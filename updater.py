import torch
import torch.optim as optim
import torch.nn.functional as F

class UnifiedConfidenceUpdater:
    def __init__(self, model, dataset, lr=0.001, gamma=0.8, device="cpu", args=None, ablation_mode="full"):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.args = args
        self.ablation_mode = ablation_mode

        # Freeze MLP and relation embeddings
        for param in self.model.mlp_mean.parameters():
            param.requires_grad = False
        for param in self.model.mlp_var.parameters():
            param.requires_grad = False
        self.model.relation_emb.weight.requires_grad = False

        # Unfreeze the last linear layer of mlp_mean so consistency regularization
        # can fine-tune the prediction head without receiving noisy pseudo-label gradients
        for param in self.model.mlp_mean[3].parameters():
            param.requires_grad = True

    def step(self, new_facts_batch):
        h_idx = torch.tensor([f[0] for f in new_facts_batch], dtype=torch.long, device=self.device)
        r_idx = torch.tensor([f[1] for f in new_facts_batch], dtype=torch.long, device=self.device)
        t_idx = torch.tensor([f[2] for f in new_facts_batch], dtype=torch.long, device=self.device)

        self._init_new_entities(h_idx, r_idx, t_idx)

        base_edge_idx, base_edge_type, base_edge_conf = self.dataset.get_base_graph_data()
        base_edge_idx = base_edge_idx.to(self.device)
        base_edge_type = base_edge_type.to(self.device)
        base_edge_conf = base_edge_conf.to(self.device)

        self.model.eval()
        with torch.no_grad():
            old_z = self.model(base_edge_idx, base_edge_type, base_edge_conf)
            mu_without, _ = self.model.predict(old_z[base_edge_idx[0]], base_edge_type, old_z[base_edge_idx[1]])

            # Save old raw embeddings for elastic anchor regularization
            old_raw_emb = self.model.entity_emb.weight.detach().clone()

        # 1. Consistency inference stage (replaces EM pseudo-label inference)
        new_mu, new_sigma_sq = self._consistency_inference(h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf)

        # 2. Causal influence assessment
        S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq)

        # 3. Bayesian belief filtering
        c_new, c_old = self._bayesian_belief_filtering(base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau)

        # 4. Local representation refinement
        real_affected_count = self._local_representation_refinement(base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu)

        self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)

        change_tensor = torch.abs(c_new - c_old)
        return new_mu.detach(), change_tensor.mean().item(), change_tensor.max().item(), real_affected_count

    def _init_new_entities(self, h_idx, r_idx, t_idx):
        with torch.no_grad():
            ent_weight = self.model.entity_emb.weight
            rel_weight = self.model.relation_emb.weight
            new_ents = self.dataset.new_entities

            # Build a tensor of new entity IDs for vectorized membership checks
            if new_ents:
                new_ents_tensor = torch.tensor(list(new_ents), dtype=torch.long, device=self.device)
            else:
                new_ents_tensor = torch.empty(0, dtype=torch.long, device=self.device)

            for ent_id in new_ents:
                msgs = []
                # As head entity: h=ent_id, neighbor is t, relation r
                mask_h = (h_idx == ent_id)
                if mask_h.any():
                    t_neighbors = t_idx[mask_h]
                    r_neighbors = r_idx[mask_h]
                    old_mask = ~torch.isin(t_neighbors, new_ents_tensor)
                    if old_mask.any():
                        msgs.append((ent_weight[t_neighbors[old_mask]] - rel_weight[r_neighbors[old_mask]]).mean(dim=0))

                # As tail entity: t=ent_id, neighbor is h, relation r
                mask_t = (t_idx == ent_id)
                if mask_t.any():
                    h_neighbors = h_idx[mask_t]
                    r_neighbors = r_idx[mask_t]
                    old_mask = ~torch.isin(h_neighbors, new_ents_tensor)
                    if old_mask.any():
                        msgs.append((ent_weight[h_neighbors[old_mask]] + rel_weight[r_neighbors[old_mask]]).mean(dim=0))

                if msgs:
                    ent_weight[ent_id] = torch.stack(msgs).mean(dim=0)
                else:
                    # Fallback: average all neighbors if no old-entity neighbors available
                    neighbors = []
                    if mask_h.any():
                        neighbors.extend(t_idx[mask_h].tolist())
                    if mask_t.any():
                        neighbors.extend(h_idx[mask_t].tolist())
                    if neighbors:
                        neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
                        ent_weight[ent_id] = ent_weight[neighbor_tensor].mean(dim=0)

    def _consistency_inference(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        """Label Propagation + Consistency Regularization + Base outer constraint."""
        self.model.train()

        # Collect trainable parameters: entity embeddings + unfrozen last layer of mlp_mean
        trainable_params = [self.model.entity_emb.weight]
        trainable_params += [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)

        # Build combined graph (base edges + new edges)
        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)

        # ------------------------------------------------------------------
        # Step 1: Build label propagation index from base graph (one-time)
        # ------------------------------------------------------------------
        # hr_conf: h as head with relation r -> [conf values]
        # tr_conf: t as tail with relation r -> [conf values]
        # r_conf:  relation r global -> [conf values]
        hr_conf = {}  # (h_id, r_id) -> [conf values]
        tr_conf = {}  # (t_id, r_id) -> [conf values]
        r_conf = {}   # r_id -> [conf values]

        base_h = base_edge_idx[0].tolist()
        base_t = base_edge_idx[1].tolist()
        base_r = base_edge_type.tolist()
        base_c = base_edge_conf.tolist()

        for i in range(len(base_h)):
            h_val = base_h[i]
            t_val = base_t[i]
            r_val = base_r[i]
            c_val = base_c[i]

            hr_key = (h_val, r_val)
            tr_key = (t_val, r_val)

            if hr_key not in hr_conf:
                hr_conf[hr_key] = []
            hr_conf[hr_key].append(c_val)

            if tr_key not in tr_conf:
                tr_conf[tr_key] = []
            tr_conf[tr_key].append(c_val)

            if r_val not in r_conf:
                r_conf[r_val] = []
            r_conf[r_val].append(c_val)

        # Pre-compute relation-level means
        r_mean = {r: sum(cs) / len(cs) for r, cs in r_conf.items()}

        # ------------------------------------------------------------------
        # Step 2: Compute individualized label-propagation pseudo-labels
        # ------------------------------------------------------------------
        lp_labels = []
        for i in range(h_idx.shape[0]):
            h_val = h_idx[i].item()
            r_val = r_idx[i].item()
            t_val = t_idx[i].item()

            s1 = hr_conf.get((h_val, r_val), None)  # h's outgoing edges with relation r
            s2 = tr_conf.get((t_val, r_val), None)  # t's incoming edges with relation r
            s3 = r_mean.get(r_val, 0.5)             # global relation prior

            s1_mean = sum(s1) / len(s1) if s1 else None
            s2_mean = sum(s2) / len(s2) if s2 else None

            if s1_mean is not None and s2_mean is not None:
                label = 0.4 * s1_mean + 0.4 * s2_mean + 0.2 * s3
            elif s1_mean is not None:
                label = 0.6 * s1_mean + 0.4 * s3
            elif s2_mean is not None:
                label = 0.6 * s2_mean + 0.4 * s3
            else:
                label = s3

            lp_labels.append(label)

        struct_prior_conf = torch.tensor(lp_labels, dtype=torch.float, device=self.device)

        # Initialize GNN input confidences for new facts using struct priors
        combined_edge_conf = torch.cat([base_edge_conf, struct_prior_conf], dim=0)

        old_emb_snapshot = self.model.entity_emb.weight.detach().clone()
        base_num_ent = self.dataset.base_num_ent

        # Snapshot unfrozen MLP parameters for anchor regularisation
        old_mlp_params = {
            name: param.detach().clone()
            for name, param in self.model.mlp_mean.named_parameters()
            if param.requires_grad
        }

        # Identify base edges that share at least one entity with the new facts
        new_ents_tensor = torch.cat([h_idx, t_idx]).unique()
        replay_mask = (
            torch.isin(base_edge_idx[0], new_ents_tensor) |
            torch.isin(base_edge_idx[1], new_ents_tensor)
        )

        em_steps = self.args.em_steps if self.args else 5
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        lambda_reg = min(lambda_reg, 0.01)
        edge_drop_rate = getattr(self.args, 'edge_drop_rate', 0.2) if self.args else 0.2
        alpha_base = getattr(self.args, 'alpha_base', 1.0) if self.args else 1.0
        mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01

        # ------------------------------------------------------------------
        # Step 2: Zero-shot GNN inference → uncertainty-adaptive target_conf
        # ------------------------------------------------------------------
        self.model.eval()
        with torch.no_grad():
            init_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            mu_zero_shot, sigma_zero_shot = self.model.predict(init_z[h_idx], r_idx, init_z[t_idx])

        # Uncertainty-adaptive blend: high-sigma facts lean on struct prior
        confidence_weight = 1.0 / (1.0 + sigma_zero_shot.detach())
        target_conf = (
            (1 - confidence_weight) * struct_prior_conf
            + confidence_weight * mu_zero_shot.detach()
        )

        self.model.train()

        mu_v1 = sigma_v1 = None

        for step in range(em_steps):
            optimizer.zero_grad()

            # === View 1: normal forward on full combined graph ===
            z_normal = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            mu_v1, sigma_v1 = self.model.predict(z_normal[h_idx], r_idx, z_normal[t_idx])

            # === View 2: edge-dropped forward (stop gradient) ===
            drop_mask = torch.rand(combined_edge_index.shape[1], device=self.device) > edge_drop_rate
            # Ensure at least one edge survives to keep the GNN well-defined
            if not drop_mask.any():
                drop_mask = drop_mask.clone()
                drop_mask[0] = True
            dropped_edge_index = combined_edge_index[:, drop_mask]
            dropped_edge_type = combined_edge_type[drop_mask]
            dropped_edge_conf = combined_edge_conf[drop_mask]

            with torch.no_grad():
                z_dropped = self.model(dropped_edge_index, dropped_edge_type, dropped_edge_conf)
                mu_v2, _ = self.model.predict(z_dropped[h_idx], r_idx, z_dropped[t_idx])

            # (1) Consistency loss: View 1 chases View 2 (detached), no GT needed
            loss_consistency = torch.mean((mu_v1 - mu_v2.detach()) ** 2)

            # (2) Heteroscedastic pseudo-label loss: structure-aware target drives learning
            loss_pseudo = torch.mean(
                ((target_conf.detach() - mu_v1) ** 2) / (sigma_v1 + 1e-6)
                + torch.log(sigma_v1 + 1e-6)
            )

            # (3) Base outer-constraint loss: use GT confidences on shared-entity base edges
            if replay_mask.any():
                replay_edges = base_edge_idx[:, replay_mask]
                replay_rels = base_edge_type[replay_mask]
                replay_conf = base_edge_conf[replay_mask]
                mu_base_pred, _ = self.model.predict(
                    z_normal[replay_edges[0]], replay_rels, z_normal[replay_edges[1]]
                )
                loss_base = F.mse_loss(mu_base_pred, replay_conf)
            else:
                loss_base = torch.tensor(0.0, device=self.device)

            # (4) Embedding anchor regularisation (prevent catastrophic forgetting)
            curr_z = self.model.entity_emb.weight
            loss_reg = lambda_reg * torch.mean(
                (curr_z[:base_num_ent] - old_emb_snapshot[:base_num_ent]) ** 2
            )

            # (5) MLP anchor regularisation (keep unfrozen MLP head close to base model)
            loss_mlp_anchor = torch.tensor(0.0, device=self.device)
            for name, param in self.model.mlp_mean.named_parameters():
                if param.requires_grad and name in old_mlp_params:
                    loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
            loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

            # Dynamic weight schedule: early steps trust pseudo-label more (direction),
            # later steps trust consistency more (stability)
            pseudo_weight = max(0.3, 1.0 - step / max(1, em_steps - 1))
            consistency_weight = 1.0 - pseudo_weight

            loss_total = (
                pseudo_weight * loss_pseudo
                + consistency_weight * loss_consistency
                + alpha_base * loss_base
                + loss_reg
                + loss_mlp_anchor
            )
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # EMA update: pseudo label gradually moves toward model prediction
            with torch.no_grad():
                momentum = 0.9 - 0.4 * (step / max(1, em_steps - 1))  # 0.9 → 0.5
                target_conf = momentum * target_conf + (1 - momentum) * mu_v1.detach()

            # Refresh edge_conf for new facts using the updated model
            with torch.no_grad():
                z_updated = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
                mu_updated, _ = self.model.predict(z_updated[h_idx], r_idx, z_updated[t_idx])
                combined_edge_conf = torch.cat([base_edge_conf, mu_updated.detach()], dim=0)

        # Fallback if em_steps == 0
        if mu_v1 is None:
            self.model.eval()
            with torch.no_grad():
                z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
                mu_v1, sigma_v1 = self.model.predict(z[h_idx], r_idx, z[t_idx])

        return mu_v1.detach(), sigma_v1.detach()

    def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq):
        self.model.eval()

        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
        combined_edge_conf = torch.cat([base_edge_conf, new_mu], dim=0)

        with torch.no_grad():
            curr_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            mu_with, sigma_sq_old = self.model.predict(curr_z[base_edge_idx[0]], base_edge_type, curr_z[base_edge_idx[1]])

            ITE = torch.abs(mu_with - mu_without)

            # Uncertainty-aware discount: unreliable interventions have less influence
            intervention_reliability = 1.0 / (1.0 + new_sigma_sq.mean().item())

            # Confounder adjustment: high-degree hub nodes are penalized
            t_nodes = base_edge_idx[1]
            node_degrees = torch.bincount(t_nodes, minlength=self.dataset.base_num_ent)
            target_degrees = node_degrees[t_nodes].float()
            confounder_penalty = 1.0 / torch.log2(target_degrees + 2.0)

            S_tau = ITE * intervention_reliability * confounder_penalty * self.gamma
            S_tau = torch.where(S_tau < 1e-4, torch.zeros_like(S_tau), S_tau)

        return S_tau, mu_with, sigma_sq_old

    def _bayesian_belief_filtering(self, base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau):
        epsilon = self.args.epsilon if self.args else 1e-4

        c_old = []
        h_list = base_edge_idx[0].cpu().numpy()
        t_list = base_edge_idx[1].cpu().numpy()
        r_list = base_edge_type.cpu().numpy()

        for i in range(len(h_list)):
            fact_tuple = (h_list[i], r_list[i], t_list[i])
            c_old.append(self.dataset.belief_state.get(fact_tuple, 0.5))

        c_old = torch.tensor(c_old, dtype=torch.float, device=self.device)

        if self.ablation_mode == "wo_bayes":
            c_new = 0.5 * c_old + 0.5 * mu_with
        else:
            w_tau = S_tau / (S_tau + sigma_sq_old + epsilon)
            c_new = (1 - w_tau) * c_old + w_tau * mu_with

        return torch.clamp(c_new, 0.0, 1.0), c_old

    def _local_representation_refinement(self, base_edge_idx, base_edge_type, c_new, old_z, S_tau, h_idx, r_idx, t_idx, new_mu):
        """Elastic rebalancing: fit new knowledge + evolve affected old knowledge + elastic anchor"""
        threshold = self.args.influence_threshold if self.args else 0.01
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        # Cap lambda_reg to prevent over-restriction in refinement stage
        lambda_reg = min(lambda_reg, 0.01)
        alpha = 1.0
        steps = self.args.refine_steps if self.args else 3

        if self.ablation_mode == "wo_causal":
            affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
        else:
            affected_mask = S_tau > threshold

        real_affected_count = affected_mask.sum().item()

        self.model.train()
        optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)

        for _ in range(steps):
            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight

            # 1. Absolute priority: ensure new fact embeddings are learned
            mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
            loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)

            # 2. Reasonable evolution: push affected old facts toward Bayesian-updated c_new
            if affected_mask.any():
                affected_edges = base_edge_idx[:, affected_mask]
                affected_rels = base_edge_type[affected_mask]
                mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]])
                loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
            else:
                loss_affected = torch.tensor(0.0, device=self.device)

            # 3. Elastic anchor: L2 distance in embedding space
            loss_reg = lambda_reg * torch.mean((curr_z[:self.dataset.base_num_ent] - old_z[:self.dataset.base_num_ent].detach()) ** 2)

            loss_total = loss_new + alpha * loss_affected + loss_reg
            loss_total.backward()
            optimizer.step()

        return real_affected_count

    def _update_dataset_belief(self, base_edge_idx, base_edge_type, updated_beliefs):
        h_list = base_edge_idx[0].cpu().numpy()
        t_list = base_edge_idx[1].cpu().numpy()
        r_list = base_edge_type.cpu().numpy()
        c_list = updated_beliefs.cpu().numpy()

        for i in range(len(h_list)):
            fact_tuple = (h_list[i], r_list[i], t_list[i])
            self.dataset.belief_state[fact_tuple] = float(c_list[i])

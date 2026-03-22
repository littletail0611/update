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

        # Save snapshot of MLP params for anchor regularization
        self.old_mlp_params = {}
        for name, param in self.model.mlp_mean.named_parameters():
            self.old_mlp_params[name] = param.detach().clone()

        # Freeze most params but allow last linear layer of mlp_mean to adapt
        for name, param in self.model.mlp_mean.named_parameters():
            if name.startswith('3.'):  # The last nn.Linear(emb_dim, 1) at Sequential index 3
                param.requires_grad = True
            else:
                param.requires_grad = False

        for param in self.model.mlp_var.parameters():
            param.requires_grad = False
        self.model.relation_emb.weight.requires_grad = False

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

        # 1. EM inference stage
        new_mu, new_sigma_sq = self._local_em_inference(h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf)

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

    def _local_em_inference(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        trainable_params = [self.model.entity_emb.weight]
        trainable_params += [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        self.model.train()
        optimizer = optim.Adam(trainable_params, lr=self.lr)

        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)

        em_steps = self.args.em_steps if self.args else 5
        lambda_reg = self.args.lambda_reg if self.args else 0.001

        old_emb_snapshot = self.model.entity_emb.weight.detach().clone()
        base_num_ent = self.dataset.base_num_ent

        # Macroscopic calibration: relation prior from base knowledge
        rel_priors = []
        for r in r_idx:
            mask = (base_edge_type == r)
            if mask.any():
                rel_priors.append(base_edge_conf[mask].mean().item())
            else:
                rel_priors.append(0.5)

        prior_conf = torch.tensor(rel_priors, dtype=torch.float, device=self.device)

        # P3: Adaptive prior/zero-shot mixing based on uncertainty
        self.model.eval()
        with torch.no_grad():
            init_combined_conf = torch.cat([base_edge_conf, prior_conf], dim=0)
            init_z = self.model(combined_edge_index, combined_edge_type, init_combined_conf)
            mu_zero_shot, sigma_zero_shot = self.model.predict(init_z[h_idx], r_idx, init_z[t_idx])

        # confidence_weight ∈ (0,1): weight applied to the zero-shot prediction.
        # When sigma_zero_shot is high (high uncertainty), confidence_weight → 0,
        # so target_conf relies more on the stable relation prior (prior_conf).
        confidence_weight = 1.0 / (1.0 + sigma_zero_shot.detach())
        target_conf = (1 - confidence_weight) * prior_conf + confidence_weight * mu_zero_shot.detach()
        self.model.train()

        # P2: Identify entities in new facts for local replay
        new_ent_set = set(h_idx.tolist() + t_idx.tolist())
        replay_mask = torch.tensor(
            [(base_edge_idx[0, i].item() in new_ent_set or base_edge_idx[1, i].item() in new_ent_set)
             for i in range(base_edge_idx.shape[1])],
            dtype=torch.bool, device=self.device
        )

        for step in range(em_steps):
            # P1: Dynamic momentum: from 0.9 down to 0.3
            curr_momentum = 0.9 - 0.6 * (step / max(1, em_steps - 1))

            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight

            combined_edge_conf = torch.cat([base_edge_conf, target_conf], dim=0)
            updated_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

            mu_pred, sigma_pred = self.model.predict(updated_z[h_idx], r_idx, updated_z[t_idx])

            with torch.no_grad():
                target_conf = curr_momentum * target_conf + (1 - curr_momentum) * mu_pred.detach()

            loss_pseudo = torch.mean(((target_conf - mu_pred) ** 2) / (sigma_pred + 1e-6) + torch.log(sigma_pred + 1e-6))

            # P2: Local replay - only edges sharing entities with new facts
            if replay_mask.any():
                replay_edges = base_edge_idx[:, replay_mask]
                replay_rels = base_edge_type[replay_mask]
                replay_conf = base_edge_conf[replay_mask]
                mu_base_pred, sigma_base_pred = self.model.predict(
                    updated_z[replay_edges[0]], replay_rels, updated_z[replay_edges[1]])
                weight = replay_conf.detach() / (sigma_base_pred.detach() + 1e-4)
                weighted_squared_error = weight * (mu_base_pred - replay_conf) ** 2
                loss_replay = torch.sum(weighted_squared_error) / (torch.sum(weight) + 1e-8)
            else:
                loss_replay = torch.tensor(0.0, device=self.device)

            loss_reg = lambda_reg * torch.mean((curr_z[:base_num_ent] - old_emb_snapshot[:base_num_ent]) ** 2)

            # P2: MLP anchor loss to protect old knowledge in MLP
            mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01
            loss_mlp_anchor = mlp_anchor_coeff * sum(
                ((p - self.old_mlp_params[n]) ** 2).mean()
                for n, p in self.model.mlp_mean.named_parameters() if p.requires_grad
            )

            # P2: Decreasing replay scale: protect old knowledge early, let new knowledge in later
            replay_scale = 2.0 - (step / max(1, em_steps - 1))
            loss_total = loss_pseudo + loss_reg + replay_scale * loss_replay + loss_mlp_anchor
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(self.model.entity_emb.parameters(), max_norm=1.0)
            optimizer.step()

        return mu_pred, sigma_pred

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
        alpha = 1.0
        steps = self.args.refine_steps if self.args else 3

        if self.ablation_mode == "wo_causal":
            affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
        else:
            affected_mask = S_tau > threshold

        real_affected_count = affected_mask.sum().item()

        self.model.train()
        trainable_params = [self.model.entity_emb.weight]
        trainable_params += [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)

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

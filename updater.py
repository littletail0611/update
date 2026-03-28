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

        # 1. Propagate-then-finetune stage (deterministic label propagation + base-GT-only fine-tuning)
        raw_new_mu, new_sigma_sq = self._propagate_then_finetune(h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf)
        new_mu = self._relation_aware_calibration(raw_new_mu, new_sigma_sq, r_idx, base_edge_type, base_edge_conf)

        # 2. Causal influence assessment
        S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq)

        # 3. Bayesian belief filtering
        c_new, c_old = self._bayesian_belief_filtering(base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau)

        # 4. Local representation refinement
        real_affected_count = self._local_representation_refinement(base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu)

        self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)

        # Write new-fact beliefs back to dataset
        new_mu_np = new_mu.detach().cpu().numpy()
        h_np = h_idx.cpu().numpy()
        r_np = r_idx.cpu().numpy()
        t_np = t_idx.cpu().numpy()
        for i in range(len(h_np)):
            fact_tuple = (int(h_np[i]), int(r_np[i]), int(t_np[i]))
            self.dataset.belief_state[fact_tuple] = float(new_mu_np[i])

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

    def _relation_aware_calibration(self, new_mu, new_sigma_sq, r_idx,
                                     base_edge_type, base_edge_conf):
        """Calibrate model predictions for new facts using relation-level priors.
        
        For each new fact (h, r, t):
            calibrated_mu = w * new_mu + (1 - w) * prior_mu_r
            where w = model_precision / (model_precision + prior_precision)
        """
        calibrated = torch.zeros_like(new_mu)
        
        # Precompute per-relation statistics from base graph
        r_stats = {}
        for r_val in base_edge_type.unique().tolist():
            mask = (base_edge_type == r_val)
            confs = base_edge_conf[mask]
            r_stats[r_val] = {
                'mean': confs.mean().item(),
                'var': confs.var().item() if confs.numel() > 1 else 0.1,
                'count': confs.numel()
            }
        
        global_mean = base_edge_conf.mean().item() if base_edge_conf.numel() > 0 else 0.5
        global_var = base_edge_conf.var().item() if base_edge_conf.numel() > 1 else 0.1
        
        for i in range(new_mu.shape[0]):
            r = r_idx[i].item()
            stats = r_stats.get(r, {'mean': global_mean, 'var': global_var, 'count': 1})
            
            prior_mu = stats['mean']
            prior_precision = 1.0 / (stats['var'] + 1e-6)
            model_precision = 1.0 / (new_sigma_sq[i].item() + 1e-6)
            
            # Bayesian precision weighting
            w = model_precision / (model_precision + prior_precision)
            calibrated[i] = w * new_mu[i] + (1 - w) * prior_mu
        
        return calibrated

    @staticmethod
    def _make_selective_grad_hook(update_mask):
        """Return a gradient hook that zeroes out gradients for entities not in update_mask.

        The incoming gradient tensor must be cloned before masking because PyTorch
        autograd forbids in-place modification of gradient leaf tensors.
        """
        def hook(grad):
            grad_clone = grad.clone()
            grad_clone[~update_mask] = 0.0
            return grad_clone
        return hook

    def _few_shot_confidence_init(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, top_k=10, tau=0.1):
        """Few-shot inspired confidence initialization for new facts.

        For each new fact (h, r, t), finds old facts with the SAME relation r that are
        semantically similar (measured by cosine similarity of the entity-pair embedding
        sum h_emb + t_emb), then aggregates their confidences via softmax-weighted sum.

        Falls back to the global mean confidence when no same-relation old facts exist.
        """
        N = h_idx.shape[0]
        propagated_conf = torch.zeros(N, dtype=torch.float, device=self.device)
        global_mean = base_edge_conf.mean() if base_edge_conf.numel() > 0 else torch.tensor(0.5, device=self.device)

        ent_weight = self.model.entity_emb.weight

        for i in range(N):
            r = r_idx[i].item()
            # Find old facts with the same relation
            same_rel_mask = (base_edge_type == r)
            if not same_rel_mask.any():
                propagated_conf[i] = global_mean
                continue

            same_rel_h = base_edge_idx[0][same_rel_mask]
            same_rel_t = base_edge_idx[1][same_rel_mask]
            same_rel_confs = base_edge_conf[same_rel_mask]

            # Semantic similarity via entity-pair embedding: L2-normalised (h+t) cosine sim
            new_feat = F.normalize(
                (ent_weight[h_idx[i]] + ent_weight[t_idx[i]]).unsqueeze(0), p=2, dim=-1
            )  # [1, D]
            old_feat = F.normalize(
                ent_weight[same_rel_h] + ent_weight[same_rel_t], p=2, dim=-1
            )  # [K, D]

            sim = torch.mm(new_feat, old_feat.t()).squeeze(0)  # [K]

            # Select top-k most semantically similar same-relation old facts
            k = min(top_k, sim.shape[0])
            topk_sim, topk_idx = torch.topk(sim, k)

            # Softmax-weighted confidence aggregation
            weights = F.softmax(topk_sim / tau, dim=0)
            propagated_conf[i] = (weights * same_rel_confs[topk_idx]).sum()

        return propagated_conf

    def _propagate_then_finetune(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        """Stage 1: Few-shot confidence init + new-entity-only fine-tuning.

        Phase 1 (no gradients):
            Initialize new-fact confidences via _few_shot_confidence_init:
            for each new fact, find same-relation old facts that are semantically
            similar (entity-pair cosine sim) and aggregate their confidences.

        Phase 2 (Stage 1 fine-tuning):
            Only train on new facts; old entity embeddings are frozen via a
            gradient hook so they receive no gradient updates.  Only new entity
            representations (IDs >= base_num_ent) are learned here.

        Returns (new_mu, new_sigma) predicted after Stage 1 fine-tuning.
        """
        finetune_steps = getattr(self.args, 'finetune_steps', 5) if self.args else 5
        mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01
        dynamic_update_interval = getattr(self.args, 'dynamic_update_interval', 2) if self.args else 2
        alpha_new = getattr(self.args, 'alpha_new_supervision', 0.3) if self.args else 0.3
        base_num_ent = self.dataset.base_num_ent

        # ------------------------------------------------------------------
        # Phase 1: Few-shot inspired confidence initialization for new facts
        # Find same-relation old facts that are semantically similar and use
        # their confidences to initialise new-fact confidence values.
        # ------------------------------------------------------------------
        with torch.no_grad():
            propagated_conf = self._few_shot_confidence_init(
                h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf
            )

        # Build combined graph using few-shot initialised confidence for new edges
        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
        combined_edge_conf = torch.cat([base_edge_conf, propagated_conf], dim=0)

        # ------------------------------------------------------------------
        # Phase 2: Stage 1 fine-tuning – only learn new entity representations
        # Old entity embeddings are frozen via a gradient hook (Stage 1 rule:
        # train only on new facts, only update new entities).
        # ------------------------------------------------------------------
        old_mlp_params = {
            name: param.detach().clone()
            for name, param in self.model.mlp_mean.named_parameters()
            if param.requires_grad
        }

        trainable_params = [self.model.entity_emb.weight]
        trainable_params += [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)

        # Gradient hook: zero out gradients for old entities so only new entity
        # representations (IDs >= base_num_ent) are updated during Stage 1.
        n_ents_s1 = self.model.entity_emb.weight.shape[0]
        stage1_update_mask = torch.zeros(n_ents_s1, dtype=torch.bool, device=self.device)
        stage1_update_mask[base_num_ent:] = True  # only new entities
        hook_s1 = self.model.entity_emb.weight.register_hook(
            self._make_selective_grad_hook(stage1_update_mask)
        )

        self.model.train()

        for step_i in range(finetune_steps):
            optimizer.zero_grad()

            z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

            # Stage 1: supervise only on new facts using few-shot initialised confidences
            mu_new_pred, _ = self.model.predict(z[h_idx], r_idx, z[t_idx])
            ramp = min(1.0, (step_i + 1) / finetune_steps)
            loss_new = alpha_new * ramp * F.mse_loss(mu_new_pred, propagated_conf.detach())

            # MLP anchor: keep prediction head close to base model
            loss_mlp_anchor = torch.tensor(0.0, device=self.device)
            for name, param in self.model.mlp_mean.named_parameters():
                if param.requires_grad and name in old_mlp_params:
                    loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
            loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

            loss_total = loss_new + loss_mlp_anchor
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Dynamic pseudo-label update: refresh combined_edge_conf for new facts
            # every dynamic_update_interval steps so the soft target tracks the model.
            if dynamic_update_interval > 0 and step_i > 0 and step_i % dynamic_update_interval == 0:
                with torch.no_grad():
                    combined_edge_conf = torch.cat(
                        [base_edge_conf, mu_new_pred.detach().clamp(0.0, 1.0)], dim=0
                    )

        hook_s1.remove()

        # Final prediction for new facts using Stage-1-fine-tuned embeddings
        self.model.eval()
        with torch.no_grad():
            z_final = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            new_mu, new_sigma = self.model.predict(z_final[h_idx], r_idx, z_final[t_idx])

        return new_mu.detach(), new_sigma.detach()

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
        """Stage 2: Jointly update new facts and affected old facts with selective entity updates.

        Only updates entity representations that appear in:
          - All new facts  (h_idx, t_idx)
          - Affected old facts  (edges with S_tau > threshold)

        Entities that are not in either set remain frozen via a gradient hook.
        Old entity representations in the affected set are regularised with an L2
        penalty against their original embeddings (old_z) to prevent excessive drift.
        """
        threshold = self.args.influence_threshold if self.args else 0.01
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        lambda_reg = min(lambda_reg, 0.01)
        func_anchor_ratio = getattr(self.args, 'func_anchor_ratio', 0.9) if self.args else 0.9
        alpha = 1.0
        steps = self.args.refine_steps if self.args else 3
        base_num_ent = self.dataset.base_num_ent

        if self.ablation_mode == "wo_causal":
            affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
        else:
            affected_mask = S_tau > threshold

        real_affected_count = affected_mask.sum().item()

        # Compute entities to update in Stage 2:
        #   = all entities in new facts  ∪  entities in affected old facts
        new_ents = torch.cat([h_idx, t_idx]).unique()
        if affected_mask.any():
            affected_edges = base_edge_idx[:, affected_mask]
            affected_old_ents = torch.cat([affected_edges[0], affected_edges[1]]).unique()
        else:
            affected_old_ents = torch.empty(0, dtype=torch.long, device=self.device)

        ents_to_update = torch.cat([new_ents, affected_old_ents]).unique()

        # Old entities in affected set (IDs < base_num_ent) – need regularisation
        if affected_old_ents.numel() > 0:
            old_ents_for_reg = affected_old_ents[affected_old_ents < base_num_ent]
        else:
            old_ents_for_reg = torch.empty(0, dtype=torch.long, device=self.device)

        # Selective gradient mask: only entities in ents_to_update receive gradients
        n_ents = self.model.entity_emb.weight.shape[0]
        update_mask = torch.zeros(n_ents, dtype=torch.bool, device=self.device)
        update_mask[ents_to_update] = True

        # Snapshot old predictions on affected edges for functional anchoring
        with torch.no_grad():
            if affected_mask.any():
                affected_edges_snap = base_edge_idx[:, affected_mask]
                affected_rels_snap = base_edge_type[affected_mask]
                mu_old_affected, _ = self.model.predict(
                    old_z[affected_edges_snap[0]], affected_rels_snap, old_z[affected_edges_snap[1]]
                )
                mu_old_affected = mu_old_affected.detach()
            else:
                mu_old_affected = None

        self.model.train()
        optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)

        # Gradient hook: only update selected entities (new + affected old)
        hook_s2 = self.model.entity_emb.weight.register_hook(
            self._make_selective_grad_hook(update_mask)
        )

        for _ in range(steps):
            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight

            # 1. New fact supervision: ensure new fact representations are learned
            mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
            loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)

            # 2. Affected old fact supervision: push toward Bayesian-updated c_new
            if affected_mask.any():
                affected_edges = base_edge_idx[:, affected_mask]
                affected_rels = base_edge_type[affected_mask]
                mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]])
                loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
            else:
                loss_affected = torch.tensor(0.0, device=self.device)

            # 3. Regularise affected old entity representations: don't deviate from original
            if old_ents_for_reg.numel() > 0:
                loss_reg_old = lambda_reg * torch.mean(
                    (curr_z[old_ents_for_reg] - old_z[old_ents_for_reg].detach()) ** 2
                )
            else:
                loss_reg_old = torch.tensor(0.0, device=self.device)

            # 4. Functional anchor on affected edges: preserve prediction behaviour
            if affected_mask.any() and mu_old_affected is not None:
                affected_edges_cur = base_edge_idx[:, affected_mask]
                affected_rels_cur = base_edge_type[affected_mask]
                mu_cur_affected, _ = self.model.predict(
                    curr_z[affected_edges_cur[0]], affected_rels_cur, curr_z[affected_edges_cur[1]]
                )
                func_loss = F.mse_loss(mu_cur_affected, mu_old_affected)
                loss_reg = lambda_reg * func_anchor_ratio * func_loss + loss_reg_old
            else:
                loss_reg = loss_reg_old

            loss_total = loss_new + alpha * loss_affected + loss_reg
            loss_total.backward()
            optimizer.step()

        hook_s2.remove()

        return real_affected_count

    def _update_dataset_belief(self, base_edge_idx, base_edge_type, updated_beliefs):
        h_list = base_edge_idx[0].cpu().numpy()
        t_list = base_edge_idx[1].cpu().numpy()
        r_list = base_edge_type.cpu().numpy()
        c_list = updated_beliefs.cpu().numpy()

        for i in range(len(h_list)):
            fact_tuple = (h_list[i], r_list[i], t_list[i])
            self.dataset.belief_state[fact_tuple] = float(c_list[i])

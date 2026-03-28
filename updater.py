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

    def _label_propagation(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, hops=2):
        """Deterministic label propagation from base GT to new facts. No model involved.

        For each new fact (h, r, t):
          1-hop : base edges where h or t appears as head or tail; same-relation edges
                  get weight 2.0, different-relation edges get weight 1.0.
          2-hop : if no 1-hop base neighbors exist, look through intermediate entities
                  reachable from h/t via the current new-facts batch; half-weight (0.5×).
          fallback: global relation-prior mean; then overall graph mean.
        """
        # Precompute relation-prior and global-mean from base GT (no model access)
        r_prior = {}
        for r_val in base_edge_type.unique().tolist():
            mask = (base_edge_type == r_val)
            r_prior[r_val] = base_edge_conf[mask].mean().item()
        global_mean = base_edge_conf.mean().item() if base_edge_conf.numel() > 0 else 0.5

        propagated = []

        for i in range(h_idx.shape[0]):
            h = h_idx[i].item()
            r = r_idx[i].item()
            t = t_idx[i].item()

            # --- 1-hop: base edges involving h or t ---
            h_mask = (base_edge_idx[0] == h) | (base_edge_idx[1] == h)
            t_mask = (base_edge_idx[0] == t) | (base_edge_idx[1] == t)

            confs, wts = [], []
            for mask in [h_mask, t_mask]:
                if mask.any():
                    nc = base_edge_conf[mask]
                    nr = base_edge_type[mask]
                    same = (nr == r).float()
                    w = 1.0 + same  # same=1.0 for matching relation → w=2.0; same=0.0 otherwise → w=1.0
                    confs.append(nc)
                    wts.append(w)

            if confs:
                all_c = torch.cat(confs)
                all_w = torch.cat(wts)
                propagated.append((all_c * all_w).sum() / all_w.sum())
                continue

            # --- 2-hop fallback: intermediate entities from the new-fact batch ---
            if hops >= 2:
                inter_ents = set()
                h_new = (h_idx == h) | (t_idx == h)
                t_new = (h_idx == t) | (t_idx == t)
                for nm in [h_new, t_new]:
                    if nm.any():
                        inter_ents.update(h_idx[nm].tolist())
                        inter_ents.update(t_idx[nm].tolist())
                inter_ents.discard(h)
                inter_ents.discard(t)

                hop2_c, hop2_w = [], []
                for e in inter_ents:
                    e_mask = (base_edge_idx[0] == e) | (base_edge_idx[1] == e)
                    if e_mask.any():
                        ec = base_edge_conf[e_mask]
                        er = base_edge_type[e_mask]
                        same = (er == r).float()
                        w = 0.5 * (1.0 + same)  # 2-hop half-weight: 0.5 for different relation, 1.0 for same relation
                        hop2_c.append(ec)
                        hop2_w.append(w)

                if hop2_c:
                    all_c = torch.cat(hop2_c)
                    all_w = torch.cat(hop2_w)
                    propagated.append((all_c * all_w).sum() / all_w.sum())
                    continue

            # --- Ultimate fallback: relation prior, then global mean ---
            prior_val = r_prior.get(r, global_mean)
            propagated.append(torch.tensor(prior_val, dtype=torch.float, device=self.device))

        return torch.stack(propagated)

    def _propagate_then_finetune(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        """Two-phase approach: deterministic label propagation then base-GT-only fine-tuning.

        Phase 1 (no gradients, no model):
            Compute propagated_conf for new edges via _label_propagation.

        Phase 2 (with gradients, GT supervision only):
            Fine-tune entity_emb (+ unfrozen mlp_mean last layer) using MSE against base GT
            on edges that share at least one entity with the new facts.
            No consistency loss, no pseudo-label loss.

        Returns (new_mu, new_sigma) predicted after fine-tuning.
        """
        propagation_hops = getattr(self.args, 'propagation_hops', 2) if self.args else 2
        finetune_steps = getattr(self.args, 'finetune_steps', 5) if self.args else 5
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        lambda_reg = min(lambda_reg, 0.01)
        alpha_base = getattr(self.args, 'alpha_base', 1.0) if self.args else 1.0
        mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01
        dynamic_update_interval = getattr(self.args, 'dynamic_update_interval', 2) if self.args else 2
        func_anchor_ratio = getattr(self.args, 'func_anchor_ratio', 0.9) if self.args else 0.9
        alpha_new = getattr(self.args, 'alpha_new_supervision', 0.3) if self.args else 0.3
        base_num_ent = self.dataset.base_num_ent

        # ------------------------------------------------------------------
        # Phase 1: Deterministic label propagation (no model, no gradients)
        # ------------------------------------------------------------------
        with torch.no_grad():
            propagated_conf = self._label_propagation(
                h_idx, r_idx, t_idx,
                base_edge_idx, base_edge_type, base_edge_conf,
                hops=propagation_hops,
            )

        # Build combined graph using propagated_conf as input confidence for new edges
        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
        combined_edge_conf = torch.cat([base_edge_conf, propagated_conf], dim=0)

        # ------------------------------------------------------------------
        # Phase 2: Base-GT-supervised fine-tuning (no pseudo-labels, no consistency loss)
        # ------------------------------------------------------------------
        # Snapshot embeddings and unfrozen MLP weights for anchor regularisation
        old_emb_snapshot = self.model.entity_emb.weight.detach().clone()
        old_mlp_params = {
            name: param.detach().clone()
            for name, param in self.model.mlp_mean.named_parameters()
            if param.requires_grad
        }

        # Base edges that share at least one entity with the new facts (replay supervision)
        new_ents_tensor = torch.cat([h_idx, t_idx]).unique()
        replay_mask = (
            torch.isin(base_edge_idx[0], new_ents_tensor) |
            torch.isin(base_edge_idx[1], new_ents_tensor)
        )

        trainable_params = [self.model.entity_emb.weight]
        trainable_params += [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)

        # Snapshot functional anchor: old model predictions on replay base edges
        # Used to preserve prediction behaviour rather than absolute embedding positions
        with torch.no_grad():
            self.model.eval()
            if replay_mask.any():
                z_old_snapshot = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
                replay_edges_snap = base_edge_idx[:, replay_mask]
                replay_rels_snap = base_edge_type[replay_mask]
                mu_old_pred_replay, _ = self.model.predict(
                    z_old_snapshot[replay_edges_snap[0]], replay_rels_snap, z_old_snapshot[replay_edges_snap[1]]
                )
                mu_old_pred_replay = mu_old_pred_replay.detach()
            else:
                mu_old_pred_replay = None

        self.model.train()

        for step_i in range(finetune_steps):
            optimizer.zero_grad()

            z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

            # Sole supervision: base GT on shared-entity edges
            if replay_mask.any():
                replay_edges = base_edge_idx[:, replay_mask]
                replay_rels = base_edge_type[replay_mask]
                replay_conf_gt = base_edge_conf[replay_mask]
                mu_base_pred, _ = self.model.predict(
                    z[replay_edges[0]], replay_rels, z[replay_edges[1]]
                )
                loss_base = F.mse_loss(mu_base_pred, replay_conf_gt)
            else:
                loss_base = torch.tensor(0.0, device=self.device)

            # Direct new-fact supervision using propagated_conf as soft labels
            mu_new_pred, _ = self.model.predict(z[h_idx], r_idx, z[t_idx])
            ramp = min(1.0, (step_i + 1) / finetune_steps)
            loss_new = alpha_new * ramp * F.mse_loss(mu_new_pred, propagated_conf.detach())

            # Embedding anchor regularisation (prevent catastrophic forgetting)
            # Functional anchoring: penalise changes in predictions on replay edges
            # rather than absolute embedding positions. A weak absolute L2 term prevents
            # degenerate solutions where embeddings drift to infinity.
            curr_emb = self.model.entity_emb.weight
            weak_l2 = lambda_reg * (1.0 - func_anchor_ratio) * torch.mean(
                (curr_emb[:base_num_ent] - old_emb_snapshot[:base_num_ent]) ** 2
            )
            if replay_mask.any() and mu_old_pred_replay is not None:
                # mu_base_pred was computed on the same replay edges just above (lines 286-289);
                # reuse it as the current predictions for the functional anchor.
                func_loss = F.mse_loss(mu_base_pred, mu_old_pred_replay)
                loss_reg = lambda_reg * func_anchor_ratio * func_loss + weak_l2
            else:
                loss_reg = weak_l2

            # MLP anchor regularisation (keep unfrozen head close to base model)
            loss_mlp_anchor = torch.tensor(0.0, device=self.device)
            for name, param in self.model.mlp_mean.named_parameters():
                if param.requires_grad and name in old_mlp_params:
                    loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
            loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

            loss_total = alpha_base * loss_base + loss_new + loss_reg + loss_mlp_anchor
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Dynamic pseudo-label update: refresh combined_edge_conf for new facts
            # every dynamic_update_interval steps so the soft target tracks the model.
            # Skip step 0 — the model hasn't been updated yet at that point.
            if dynamic_update_interval > 0 and step_i > 0 and step_i % dynamic_update_interval == 0:
                with torch.no_grad():
                    combined_edge_conf = torch.cat(
                        [base_edge_conf, mu_new_pred.detach().clamp(0.0, 1.0)], dim=0
                    )

        # Final prediction for new facts using fine-tuned embeddings
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
        """Elastic rebalancing: fit new knowledge + evolve affected old knowledge + elastic anchor"""
        threshold = self.args.influence_threshold if self.args else 0.01
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        # Cap lambda_reg to prevent over-restriction in refinement stage
        lambda_reg = min(lambda_reg, 0.01)
        func_anchor_ratio = getattr(self.args, 'func_anchor_ratio', 0.9) if self.args else 0.9
        alpha = 1.0
        steps = self.args.refine_steps if self.args else 3

        if self.ablation_mode == "wo_causal":
            affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
        else:
            affected_mask = S_tau > threshold

        real_affected_count = affected_mask.sum().item()

        # Snapshot functional anchor: old predictions on affected base edges using old embeddings
        with torch.no_grad():
            if affected_mask.any():
                affected_edges_old = base_edge_idx[:, affected_mask]
                affected_rels_old = base_edge_type[affected_mask]
                mu_old_affected, _ = self.model.predict(
                    old_z[affected_edges_old[0]], affected_rels_old, old_z[affected_edges_old[1]]
                )
                mu_old_affected = mu_old_affected.detach()
            else:
                mu_old_affected = None

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

            # 3. Elastic anchor: functional anchoring preserves prediction behaviour on
            # affected edges; a weak absolute L2 prevents unbounded embedding drift.
            weak_l2 = lambda_reg * (1.0 - func_anchor_ratio) * torch.mean(
                (curr_z[:self.dataset.base_num_ent] - old_z[:self.dataset.base_num_ent].detach()) ** 2
            )
            if affected_mask.any() and mu_old_affected is not None:
                affected_edges_cur = base_edge_idx[:, affected_mask]
                affected_rels_cur = base_edge_type[affected_mask]
                mu_cur_affected, _ = self.model.predict(
                    curr_z[affected_edges_cur[0]], affected_rels_cur, curr_z[affected_edges_cur[1]]
                )
                func_loss = F.mse_loss(mu_cur_affected, mu_old_affected)
                loss_reg = lambda_reg * func_anchor_ratio * func_loss + weak_l2
            else:
                loss_reg = weak_l2

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

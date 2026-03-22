# import torch
# import torch.optim as optim
# import torch.nn.functional as F

# class UnifiedConfidenceUpdater:
#     def __init__(self, model, dataset, lr=0.001, gamma=0.8, device="cpu", args=None, ablation_mode="full"):
#         self.model = model
#         self.dataset = dataset
#         self.lr = lr
#         self.gamma = gamma
#         self.device = device
#         self.args = args
#         self.ablation_mode = ablation_mode
        
#         # 维持底层规则（MLP 和 关系投影）绝对冻结
#         for param in self.model.mlp_mean.parameters():
#             param.requires_grad = False
#         for param in self.model.mlp_var.parameters():
#             param.requires_grad = False
#         self.model.relation_emb.weight.requires_grad = False

#     def step(self, new_facts_batch):
#         h_idx = torch.tensor([f[0] for f in new_facts_batch], dtype=torch.long, device=self.device)
#         r_idx = torch.tensor([f[1] for f in new_facts_batch], dtype=torch.long, device=self.device)
#         t_idx = torch.tensor([f[2] for f in new_facts_batch], dtype=torch.long, device=self.device)
        
#         self._init_new_entities(h_idx, t_idx)
#         # self._init_new_entities(new_facts_batch)
        
#         base_edge_idx, base_edge_type, base_edge_conf = self.dataset.get_base_graph_data()
#         base_edge_idx = base_edge_idx.to(self.device)
#         base_edge_type = base_edge_type.to(self.device)
#         base_edge_conf = base_edge_conf.to(self.device)
        
#         self.model.eval()
#         with torch.no_grad():
#             old_z = self.model(base_edge_idx, base_edge_type, base_edge_conf)
#             mu_without, _ = self.model.predict(old_z[base_edge_idx[0]], base_edge_type, old_z[base_edge_idx[1]])
            
#             # ⭐ 核心修复：保存旧的原始 Embedding，用于后续的弹性锚点对比
#             old_raw_emb = self.model.entity_emb.weight.detach().clone()

#         # 1. EM 阶段
#         new_mu, new_sigma_sq = self._local_em_inference(h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf)

#         # 2. 因果影响评估
#         S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq)

#         # 3. 贝叶斯信念融合
#         c_new, c_old = self._bayesian_belief_filtering(base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau)

#         # 4. 局部表征微调 ⭐ 传入 old_raw_emb 而不是 old_z
#         real_affected_count = self._local_representation_refinement(base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu)

#         self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)
        
#         change_tensor = torch.abs(c_new - c_old)
#         return new_mu.detach(), change_tensor.mean().item(), change_tensor.max().item(), real_affected_count

#     def _init_new_entities(self, h_idx, t_idx):
#         with torch.no_grad():
#             ent_weight = self.model.entity_emb.weight
#             new_ents = self.dataset.new_entities
            
#             for ent_id in new_ents:
#                 neighbors = []
#                 mask_h = (h_idx == ent_id)
#                 mask_t = (t_idx == ent_id)
                
#                 neighbors.extend(t_idx[mask_h].tolist())
#                 neighbors.extend(h_idx[mask_t].tolist())
                
#                 if neighbors:
#                     neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
#                     avg_emb = ent_weight[neighbor_tensor].mean(dim=0)
#                     ent_weight[ent_id] = avg_emb

#     def _local_em_inference(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
#         self.model.train()
#         optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)
        
#         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
#         combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
#         combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)

#         em_steps = self.args.em_steps if self.args else 5
#         lambda_reg = self.args.lambda_reg if self.args else 0.001
        
#         old_emb_snapshot = self.model.entity_emb.weight.detach().clone()
#         base_num_ent = self.dataset.base_num_ent
        
#         # 1. 宏观校准：只看旧知识中该关系的平均置信度
#         rel_priors = []
#         for r in r_idx:
#             mask = (base_edge_type == r)
#             if mask.any():
#                 rel_priors.append(base_edge_conf[mask].mean().item())
#             else:
#                 rel_priors.append(0.5) 
                
#         prior_conf = torch.tensor(rel_priors, dtype=torch.float, device=self.device)
        
#         # 2. 微观校准：依靠 GNN 跑一次零样本前向，获取拓扑推断
#         self.model.eval()
#         with torch.no_grad():
#             init_combined_conf = torch.cat([base_edge_conf, prior_conf], dim=0)
#             init_z = self.model(combined_edge_index, combined_edge_type, init_combined_conf)
#             mu_zero_shot, _ = self.model.predict(init_z[h_idx], r_idx, init_z[t_idx])
            
#         target_conf = 0.5 * prior_conf + 0.5 * mu_zero_shot.detach()
#         self.model.train()
        
#         momentum = 0.8 
        
#         for step in range(em_steps):
#             optimizer.zero_grad()
#             curr_z = self.model.entity_emb.weight
            
#             # 使用 target_conf 作为输入，让 GNN 带着这个“初步信念”去聚合邻居
#             combined_edge_conf = torch.cat([base_edge_conf, target_conf], dim=0)
#             updated_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            
#             mu_pred, sigma_pred = self.model.predict(updated_z[h_idx], r_idx, updated_z[t_idx])
            
#             # EMA 缓慢放权：稳扎稳打
#             with torch.no_grad():
#                 target_conf = momentum * target_conf + (1.0 - momentum) * mu_pred.detach()
            
#             # (1) 异方差伪标签损失：只拟合自己，绝不与其他不确定的新事实乱攀比
#             loss_pseudo = torch.mean(((target_conf - mu_pred) ** 2) / (sigma_pred + 1e-6) + torch.log(sigma_pred + 1e-6))
            
#             # (2) 旧知识保护机制 (Replay Loss)：方差越大，惩罚越小
#             mu_base_pred, sigma_base_pred = self.model.predict(updated_z[base_edge_idx[0]], base_edge_type, updated_z[base_edge_idx[1]])
#             weight = base_edge_conf.detach() / (sigma_base_pred.detach() + 1e-4)
#             weighted_squared_error = weight * (mu_base_pred - base_edge_conf) ** 2
#             loss_replay = torch.sum(weighted_squared_error) / (torch.sum(weight) + 1e-8)
            
#             # (3) 弹性锚点正则：绝对物理限制
#             loss_reg = lambda_reg * torch.mean((curr_z[:base_num_ent] - old_emb_snapshot[:base_num_ent]) ** 2)
            
#             # 动态 Replay 权重：让旧知识的保护墙随着迭代逐渐变厚
#             replay_scale = 1.0 + (step / max(1, em_steps - 1)) 
            
#             # 彻底去掉 alpha_cl * loss_cl
#             loss_total = loss_pseudo + loss_reg + replay_scale * loss_replay
#             loss_total.backward()
            
#             torch.nn.utils.clip_grad_norm_(self.model.entity_emb.parameters(), max_norm=1.0)
#             optimizer.step()
            
#         return mu_pred, sigma_pred

#     # def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu):
#     #     self.model.eval()
#     #     with torch.no_grad():
#     #         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
#     #         combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
#     #         combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
#     #         combined_edge_conf = torch.cat([base_edge_conf, new_mu], dim=0)
            
#     #         curr_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
#     #         mu_with, sigma_sq_old = self.model.predict(curr_z[base_edge_idx[0]], base_edge_type, curr_z[base_edge_idx[1]])
            
#     #         I_tau = torch.abs(mu_with - mu_without)
#     #         S_tau = I_tau * self.gamma 
            
#     #     return S_tau, mu_with, sigma_sq_old

#     def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq):
#         self.model.eval()
        
#         # 1. 准备反事实干预图谱 (Interventional Graph)
#         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
#         combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
#         combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
#         combined_edge_conf = torch.cat([base_edge_conf, new_mu], dim=0)
        
#         with torch.no_grad():
#             # ==========================================
#             # 🌟 进阶 1：计算反事实干预下的预测 (Counterfactual Prediction)
#             # 我们直接使用当前网络状态进行前向，此时的差异完全来源于 do(Edge=1) 的干预
#             # ==========================================
#             curr_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
#             mu_with, sigma_sq_old = self.model.predict(curr_z[base_edge_idx[0]], base_edge_type, curr_z[base_edge_idx[1]])
            
#             # 基础观测因果效应 (Individual Treatment Effect, ITE)
#             ITE = torch.abs(mu_with - mu_without)
            
#             # ==========================================
#             # 🌟 进阶 2：异方差干预强度折扣 (Uncertainty-Aware Discount)
#             # 如果干预源（新事实）极度不确定，我们不能盲目相信它带来的影响
#             # ==========================================
#             # 将新事实的方差扩展/池化，代表这次"干预"的噪音水平
#             # 这里的 1.0 是一个平滑项，防止除以极小值
#             intervention_reliability = 1.0 / (1.0 + new_sigma_sq.mean().item()) 
            
#             # ==========================================
#             # 🌟 进阶 3：图结构混淆因子校准 (Confounder Adjustment via PageRank/Degree)
#             # ==========================================
#             # 计算受影响目标节点 (Target Nodes) 在 Base 图谱中的度数
#             # 目标节点通常是旧边的两端
#             t_nodes = base_edge_idx[1]
#             # 简单计算度数频率作为混淆因子代理 (或者用你 dataset 里预先算好的度数)
#             node_degrees = torch.bincount(t_nodes, minlength=self.dataset.base_num_ent)
#             target_degrees = node_degrees[t_nodes].float()
            
#             # 倾向性得分截断 (Propensity Score Truncation)
#             # 目标节点的度数越高，其特征发生改变就越有可能是因为它是"大众脸"，而不是真实的因果关联
#             # 利用 log 衰减惩罚 Hub 节点的虚假因果
#             confounder_penalty = 1.0 / torch.log2(target_degrees + 2.0) 
            
#             # ==========================================
#             # 🌟 最终计算：校准后的真实因果影响 (Adjusted Causal Influence)
#             # ==========================================
#             S_tau = ITE * intervention_reliability * confounder_penalty * self.gamma
            
#             # 增加一个物理截断机制，过滤掉由于浮点误差引起的极微小震荡
#             S_tau = torch.where(S_tau < 1e-4, torch.zeros_like(S_tau), S_tau)
            
#         return S_tau, mu_with, sigma_sq_old

#     def _bayesian_belief_filtering(self, base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau):
#         epsilon = self.args.epsilon if self.args else 1e-4
        
#         c_old = []
#         h_list = base_edge_idx[0].cpu().numpy()
#         t_list = base_edge_idx[1].cpu().numpy()
#         r_list = base_edge_type.cpu().numpy()
        
#         for i in range(len(h_list)):
#             fact_tuple = (h_list[i], r_list[i], t_list[i])
#             c_old.append(self.dataset.belief_state.get(fact_tuple, 0.5))
            
#         c_old = torch.tensor(c_old, dtype=torch.float, device=self.device)
        
#         if self.ablation_mode == "wo_bayes":
#             c_new = 0.5 * c_old + 0.5 * mu_with
#         else:
#             w_tau = S_tau / (S_tau + sigma_sq_old + epsilon)
#             c_new = (1 - w_tau) * c_old + w_tau * mu_with
            
#         return torch.clamp(c_new, 0.0, 1.0), c_old

#     def _local_representation_refinement(self, base_edge_idx, base_edge_type, c_new, old_z, S_tau, h_idx, r_idx, t_idx, new_mu):
#         """废除全局KD，启用天平重构：绝对新知识拟合 + 合理旧知识演变 + 弹性先验约束"""
#         threshold = self.args.influence_threshold if self.args else 0.01
#         lambda_reg = self.args.lambda_reg if self.args else 0.001 
#         alpha = 1.0  
#         steps = self.args.refine_steps if self.args else 3
        
#         if self.ablation_mode == "wo_causal":
#             affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
#         else:
#             affected_mask = S_tau > threshold
            
#         real_affected_count = affected_mask.sum().item()
        
#         self.model.train()
#         optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)
        
#         for _ in range(steps):
#             optimizer.zero_grad()
#             curr_z = self.model.entity_emb.weight
            
#             # 1. 绝对优先：确保新事实嵌入到位
#             mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
#             loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)
            
#             # 2. 合理演变：让受影响旧事实向贝叶斯更新后的 c_new 靠拢
#             if affected_mask.any():
#                 affected_edges = base_edge_idx[:, affected_mask]
#                 affected_rels = base_edge_type[affected_mask]
#                 mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]]) 
                
#                 loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
#             else:
#                 loss_affected = torch.tensor(0.0, device=self.device)

#             # 3. 弹性锚点：用物理空间的欧氏距离代替繁琐的输出层 KD
#             # 注意 old_z 必须 detach，防止错误回传
#             loss_reg = lambda_reg * torch.mean((curr_z[:self.dataset.base_num_ent] - old_z[:self.dataset.base_num_ent].detach()) ** 2)

#             loss_total = loss_new + alpha * loss_affected + loss_reg
#             loss_total.backward()
#             optimizer.step()
            
#         return real_affected_count

#     def _update_dataset_belief(self, base_edge_idx, base_edge_type, updated_beliefs):
#         h_list = base_edge_idx[0].cpu().numpy()
#         t_list = base_edge_idx[1].cpu().numpy()
#         r_list = base_edge_type.cpu().numpy()
#         c_list = updated_beliefs.cpu().numpy()
        
#         for i in range(len(h_list)):
#             fact_tuple = (h_list[i], r_list[i], t_list[i])
#             self.dataset.belief_state[fact_tuple] = float(c_list[i])

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
        
        # 维持底层规则（MLP 和 关系投影）绝对冻结
        for param in self.model.mlp_mean.parameters():
            param.requires_grad = False
        for param in self.model.mlp_var.parameters():
            param.requires_grad = False
        self.model.relation_emb.weight.requires_grad = False

    def step(self, new_facts_batch):
        h_idx = torch.tensor([f[0] for f in new_facts_batch], dtype=torch.long, device=self.device)
        r_idx = torch.tensor([f[1] for f in new_facts_batch], dtype=torch.long, device=self.device)
        t_idx = torch.tensor([f[2] for f in new_facts_batch], dtype=torch.long, device=self.device)
        
        self._init_new_entities(h_idx, t_idx)
        # self._init_new_entities(new_facts_batch)
        
        base_edge_idx, base_edge_type, base_edge_conf = self.dataset.get_base_graph_data()
        base_edge_idx = base_edge_idx.to(self.device)
        base_edge_type = base_edge_type.to(self.device)
        base_edge_conf = base_edge_conf.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            old_z = self.model(base_edge_idx, base_edge_type, base_edge_conf)
            mu_without, _ = self.model.predict(old_z[base_edge_idx[0]], base_edge_type, old_z[base_edge_idx[1]])
            
            # ⭐ 核心修复：保存旧的原始 Embedding，用于后续的弹性锚点对比
            old_raw_emb = self.model.entity_emb.weight.detach().clone()

        # 1. EM 阶段
        new_mu, new_sigma_sq = self._local_em_inference(h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf)

        # 2. 因果影响评估
        S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq)

        # 3. 贝叶斯信念融合
        c_new, c_old = self._bayesian_belief_filtering(base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau)

        # 4. 局部表征微调 ⭐ 传入 old_raw_emb 而不是 old_z
        real_affected_count = self._local_representation_refinement(base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu)

        self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)
        
        change_tensor = torch.abs(c_new - c_old)
        return new_mu.detach(), change_tensor.mean().item(), change_tensor.max().item(), real_affected_count

    def _init_new_entities(self, h_idx, t_idx):
        with torch.no_grad():
            ent_weight = self.model.entity_emb.weight
            new_ents = self.dataset.new_entities
            
            for ent_id in new_ents:
                neighbors = []
                mask_h = (h_idx == ent_id)
                mask_t = (t_idx == ent_id)
                
                neighbors.extend(t_idx[mask_h].tolist())
                neighbors.extend(h_idx[mask_t].tolist())
                
                if neighbors:
                    neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
                    avg_emb = ent_weight[neighbor_tensor].mean(dim=0)
                    ent_weight[ent_id] = avg_emb

    def _local_em_inference(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        self.model.train()
        optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)
        
        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)

        em_steps = self.args.em_steps if self.args else 5
        lambda_reg = self.args.lambda_reg if self.args else 0.001
        
        old_emb_snapshot = self.model.entity_emb.weight.detach().clone()
        base_num_ent = self.dataset.base_num_ent
        
        # 1. 宏观校准：只看旧知识中该关系的平均置信度
        rel_priors = []
        for r in r_idx:
            mask = (base_edge_type == r)
            if mask.any():
                rel_priors.append(base_edge_conf[mask].mean().item())
            else:
                rel_priors.append(0.5) 
                
        prior_conf = torch.tensor(rel_priors, dtype=torch.float, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            init_combined_conf = torch.cat([base_edge_conf, prior_conf], dim=0)
            init_z = self.model(combined_edge_index, combined_edge_type, init_combined_conf)
            mu_zero_shot, _ = self.model.predict(init_z[h_idx], r_idx, init_z[t_idx])
            
        target_conf = 0.6 * prior_conf + 0.4 * mu_zero_shot.detach()
        self.model.train()
        
        momentum = 0.8 
        
        for step in range(em_steps):
            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight
            
            # 使用 target_conf 作为输入，让 GNN 带着这个“初步信念”去聚合邻居
            combined_edge_conf = torch.cat([base_edge_conf, target_conf], dim=0)
            updated_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            
            mu_pred, sigma_pred = self.model.predict(updated_z[h_idx], r_idx, updated_z[t_idx])

            with torch.no_grad():
                teacher_conf = momentum * target_conf + (1.0 - momentum) * mu_pred.detach()
                target_conf = 0.9 * teacher_conf + 0.1 * prior_conf

            loss_pseudo = torch.mean(((target_conf - mu_pred) ** 2) / (sigma_pred + 1e-6) + torch.log(sigma_pred + 1e-6))

            mu_base_pred, sigma_base_pred = self.model.predict(updated_z[base_edge_idx[0]], base_edge_type, updated_z[base_edge_idx[1]])
            weight = base_edge_conf.detach() / (sigma_base_pred.detach() + 1e-4)
            weighted_squared_error = weight * (mu_base_pred - base_edge_conf) ** 2
            loss_replay = torch.sum(weighted_squared_error) / (torch.sum(weight) + 1e-8)
            
            loss_reg = lambda_reg * torch.mean((curr_z[:base_num_ent] - old_emb_snapshot[:base_num_ent]) ** 2)

            replay_scale = 1.0 + (step / max(1, em_steps - 1))
            loss_total = loss_pseudo + loss_reg + replay_scale * loss_replay
            loss_total.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.entity_emb.parameters(), max_norm=1.0)
            optimizer.step()
            
        return mu_pred, sigma_pred

    # def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu):
    #     self.model.eval()
    #     with torch.no_grad():
    #         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
    #         combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
    #         combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
    #         combined_edge_conf = torch.cat([base_edge_conf, new_mu], dim=0)
            
    #         curr_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
    #         mu_with, sigma_sq_old = self.model.predict(curr_z[base_edge_idx[0]], base_edge_type, curr_z[base_edge_idx[1]])
            
    #         I_tau = torch.abs(mu_with - mu_without)
    #         S_tau = I_tau * self.gamma 
            
    #     return S_tau, mu_with, sigma_sq_old

    def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq):
        self.model.eval()
        
        # 1. 准备反事实干预图谱 (Interventional Graph)
        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
        combined_edge_conf = torch.cat([base_edge_conf, new_mu], dim=0)
        
        with torch.no_grad():
            # ==========================================
            # 🌟 进阶 1：计算反事实干预下的预测 (Counterfactual Prediction)
            # 我们直接使用当前网络状态进行前向，此时的差异完全来源于 do(Edge=1) 的干预
            # ==========================================
            curr_z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            mu_with, sigma_sq_old = self.model.predict(curr_z[base_edge_idx[0]], base_edge_type, curr_z[base_edge_idx[1]])
            
            # 基础观测因果效应 (Individual Treatment Effect, ITE)
            ITE = torch.abs(mu_with - mu_without)
            
            # ==========================================
            # 🌟 进阶 2：异方差干预强度折扣 (Uncertainty-Aware Discount)
            # 如果干预源（新事实）极度不确定，我们不能盲目相信它带来的影响
            # ==========================================
            # 将新事实的方差扩展/池化，代表这次"干预"的噪音水平
            # 这里的 1.0 是一个平滑项，防止除以极小值
            intervention_reliability = 1.0 / (1.0 + new_sigma_sq.mean().item()) 
            
            # ==========================================
            # 🌟 进阶 3：图结构混淆因子校准 (Confounder Adjustment via PageRank/Degree)
            # ==========================================
            # 计算受影响目标节点 (Target Nodes) 在 Base 图谱中的度数
            # 目标节点通常是旧边的两端
            t_nodes = base_edge_idx[1]
            # 简单计算度数频率作为混淆因子代理 (或者用你 dataset 里预先算好的度数)
            node_degrees = torch.bincount(t_nodes, minlength=self.dataset.base_num_ent)
            target_degrees = node_degrees[t_nodes].float()
            
            # 倾向性得分截断 (Propensity Score Truncation)
            # 目标节点的度数越高，其特征发生改变就越有可能是因为它是"大众脸"，而不是真实的因果关联
            # 利用 log 衰减惩罚 Hub 节点的虚假因果
            confounder_penalty = 1.0 / torch.log2(target_degrees + 2.0) 
            
            # ==========================================
            # 🌟 最终计算：校准后的真实因果影响 (Adjusted Causal Influence)
            # ==========================================
            S_tau = ITE * intervention_reliability * confounder_penalty * self.gamma
            
            # 增加一个物理截断机制，过滤掉由于浮点误差引起的极微小震荡
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
        """废除全局KD，启用天平重构：绝对新知识拟合 + 合理旧知识演变 + 弹性先验约束"""
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
        optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight
            
            # 1. 绝对优先：确保新事实嵌入到位
            mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
            loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)
            
            # 2. 合理演变：让受影响旧事实向贝叶斯更新后的 c_new 靠拢
            if affected_mask.any():
                affected_edges = base_edge_idx[:, affected_mask]
                affected_rels = base_edge_type[affected_mask]
                mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]]) 
                
                loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
            else:
                loss_affected = torch.tensor(0.0, device=self.device)

            # 3. 弹性锚点：用物理空间的欧氏距离代替繁琐的输出层 KD
            # 注意 old_z 必须 detach，防止错误回传
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

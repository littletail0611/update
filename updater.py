# import torch
# import torch.optim as optim
# import torch.nn.functional as F

# class UnifiedConfidenceUpdater:
#     """
#     动态更新知识图谱中事实的置信度得分。
    
#     该更新器通过结合拓扑标签传播、课程微调、K-hop因果影响评估和局部的贝叶斯表征精炼等流水线，
#     来稳健地处理新事实（以及知识库外/OOKB实体）的插入，并极力避免灾难性遗忘。
#     """
    
#     def __init__(self, model, dataset, lr=0.001, gamma=0.8, device="cpu", args=None, ablation_mode="full"):
#         self.model = model
#         self.dataset = dataset
#         self.lr = lr
#         self.gamma = gamma
#         self.device = device
#         self.args = args
#         self.ablation_mode = ablation_mode

#         # 冻结 MLP 和关系嵌入以保留全局图结构
#         for param in self.model.mlp_mean.parameters():
#             param.requires_grad = False
#         for param in self.model.mlp_var.parameters():
#             param.requires_grad = False
#         self.model.relation_emb.weight.requires_grad = False

#         # # 解冻 mlp_mean 的最后线性分类层，使得模型在一致性正则化期间可以微调预测头，而不会向底层表征传递带有噪声的伪标签梯度
#         # for param in self.model.mlp_mean[3].parameters():
#         #     param.requires_grad = True

#         # 不要只解冻 [3]，解冻最后两个 block 或者全部解冻
#         for name, param in self.model.mlp_mean.named_parameters():
#             param.requires_grad = False # 全部解冻，交由 mlp_anchor_coeff 去约束

#     def step(self, new_facts_batch):
#         """
#         执行单步更新，处理一个批次的增量新事实。

#         参数:
#             new_facts_batch (list): 包含 (h, r, t) 或 (h, r, t, conf) 元组的列表。

#         返回:
#             tuple: (新事实的最终预测 mu, 置信度平均变化量, 置信度最大变化量, 受影响旧事实的数量)
#         """
#         # 解析输入数据
#         h_idx = torch.tensor([f[0] for f in new_facts_batch], dtype=torch.long, device=self.device)
#         r_idx = torch.tensor([f[1] for f in new_facts_batch], dtype=torch.long, device=self.device)
#         t_idx = torch.tensor([f[2] for f in new_facts_batch], dtype=torch.long, device=self.device)

#         has_label_list = [len(f) > 3 and f[3] is not None for f in new_facts_batch]
#         has_label_mask = torch.tensor(has_label_list, dtype=torch.bool, device=self.device)
#         known_conf_list = [float(f[3]) if (len(f) > 3 and f[3] is not None) else 0.0 for f in new_facts_batch]
#         known_conf = torch.tensor(known_conf_list, dtype=torch.float, device=self.device)

#         # 提取基础图谱数据
#         base_edge_idx, base_edge_type, base_edge_conf = self.dataset.get_base_graph_data()
#         base_edge_idx = base_edge_idx.to(self.device)
#         base_edge_type = base_edge_type.to(self.device)
#         base_edge_conf = base_edge_conf.to(self.device)

#         # 阶段 0: 针对纯新实体的安全几何初始化
#         self._init_new_entities(h_idx, r_idx, t_idx)

#         # 快照保存基础模型的预测输出和原始嵌入
#         self.model.eval()
#         with torch.no_grad():
#             old_z = self.model(base_edge_idx, base_edge_type, base_edge_conf)
#             mu_without, _ = self.model.predict(old_z[base_edge_idx[0]], base_edge_type, old_z[base_edge_idx[1]])
#             old_raw_emb = self.model.entity_emb.weight.detach().clone()

#         # 阶段 1: 两阶段防过拟合课程微调 (立锚与扩散)
#         raw_new_mu, new_sigma_sq = self._propagate_then_finetune(
#             h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, has_label_mask, known_conf
#         )
#         # 🌟 屏蔽这段贝叶斯校准！不要让它用全局均值抹杀掉我们的精细预测！
#         # new_mu = self._relation_aware_calibration(raw_new_mu, new_sigma_sq, r_idx, base_edge_type, base_edge_conf, has_label_mask, known_conf)
        
#         # 🌟 直接使用原始预测值往下走：
#         new_mu = raw_new_mu

#         # 阶段 2: 基于 K-hop 子图的因果推断评估
#         S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(
#             base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq, has_label_mask
#         )

#         # 阶段 3: 贝叶斯信念滤波
#         c_new, c_old = self._bayesian_belief_filtering(
#             base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau
#         )

#         # 阶段 4: 局部表征精炼
#         real_affected_count = self._local_representation_refinement(
#             base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu
#         )

#         # 更新数据集的全局状态
#         self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)

#         # 将新事实的置信度写回字典
#         new_mu_np = new_mu.detach().cpu().numpy()
#         known_conf_np = known_conf.cpu().numpy()
#         has_label_np = has_label_mask.cpu().numpy()
#         h_np = h_idx.cpu().numpy()
#         r_np = r_idx.cpu().numpy()
#         t_np = t_idx.cpu().numpy()
        
#         for i in range(len(h_np)):
#             fact_tuple = (int(h_np[i]), int(r_np[i]), int(t_np[i]))
#             self.dataset.belief_state[fact_tuple] = float(known_conf_np[i]) if has_label_np[i] else float(new_mu_np[i])

#         change_tensor = torch.abs(c_new - c_old)
#         return new_mu.detach(), change_tensor.mean().item(), change_tensor.max().item(), real_affected_count

#     def _init_new_entities(self, h_idx, r_idx, t_idx):
#         """
#         使用几何均值初始化知识库外（OOKB）新实体的嵌入表征。
#         抛弃关系向量加减法，完全基于邻居实体的物理坐标中心进行初始化，以防空间错位。
#         """
#         with torch.no_grad():
#             ent_weight = self.model.entity_emb.weight
#             new_ents = self.dataset.new_entities

#             if not new_ents:
#                 return

#             new_ents_tensor = torch.tensor(list(new_ents), dtype=torch.long, device=self.device)

#             for ent_id in new_ents:
#                 msgs = []
                
#                 # 当新实体作为头实体时，提取有效尾实体邻居的表征
#                 mask_h = (h_idx == ent_id)
#                 if mask_h.any():
#                     t_neighbors = t_idx[mask_h]
#                     old_mask = ~torch.isin(t_neighbors, new_ents_tensor)
#                     if old_mask.any():
#                         msgs.append(ent_weight[t_neighbors[old_mask]].mean(dim=0))

#                 # 当新实体作为尾实体时，提取有效头实体邻居的表征
#                 mask_t = (t_idx == ent_id)
#                 if mask_t.any():
#                     h_neighbors = h_idx[mask_t]
#                     old_mask = ~torch.isin(h_neighbors, new_ents_tensor)
#                     if old_mask.any():
#                         msgs.append(ent_weight[h_neighbors[old_mask]].mean(dim=0))

#                 # 聚合表征，如果不存在已知旧邻居，则回退到未验证的同批次邻居
#                 if msgs:
#                     ent_weight[ent_id] = torch.stack(msgs).mean(dim=0)
#                 else:
#                     neighbors = []
#                     if mask_h.any():
#                         neighbors.extend(t_idx[mask_h].tolist())
#                     if mask_t.any():
#                         neighbors.extend(h_idx[mask_t].tolist())
#                     if neighbors:
#                         neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
#                         ent_weight[ent_id] = ent_weight[neighbor_tensor].mean(dim=0)

#     def _relation_aware_calibration(self, new_mu, new_sigma_sq, r_idx, base_edge_type, base_edge_conf, has_label_mask=None, known_conf=None):
#         """
#         利用关系级别的贝叶斯先验来校准模型对新事实的预测结果。
#         """
#         calibrated = torch.zeros_like(new_mu)
#         r_stats = {}
        
#         # 预计算关系级别的统计分布
#         for r_val in base_edge_type.unique().tolist():
#             mask = (base_edge_type == r_val)
#             confs = base_edge_conf[mask]
#             r_stats[r_val] = {
#                 'mean': confs.mean().item(),
#                 'var': confs.var().item() if confs.numel() > 1 else 0.1,
#                 'count': confs.numel()
#             }
        
#         global_mean = base_edge_conf.mean().item() if base_edge_conf.numel() > 0 else 0.5
#         global_var = base_edge_conf.var().item() if base_edge_conf.numel() > 1 else 0.1
        
#         for i in range(new_mu.shape[0]):
#             # 绝对信任有真实标签的数据，跳过校准
#             if has_label_mask is not None and has_label_mask[i]:
#                 calibrated[i] = known_conf[i]
#                 continue

#             # 对无标签数据进行贝叶斯精度加权平滑
#             r = r_idx[i].item()
#             stats = r_stats.get(r, {'mean': global_mean, 'var': global_var, 'count': 1})
            
#             prior_mu = stats['mean']
#             prior_precision = 1.0 / (stats['var'] + 1e-6)
#             model_precision = 1.0 / (new_sigma_sq[i].item() + 1e-6)
            
#             w = model_precision / (model_precision + prior_precision)
#             calibrated[i] = w * new_mu[i] + (1 - w) * prior_mu
        
#         return calibrated

#     @staticmethod
#     def _make_selective_grad_hook(update_mask):
#         """返回一个梯度截断钩子，强制将掩码外的实体梯度置零，实现局部隔离微调。"""
#         def hook(grad):
#             grad_clone = grad.clone()
#             grad_clone[~update_mask] = 0.0
#             return grad_clone
#         return hook

#     def _topological_confidence_propagation(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
#         """
#         利用 1-hop 图拓扑随机游走，将先验置信度传播给无标签的新事实。
#         该方法取代了冷启动阶段极不稳定的“基于全局特征相似度的检索”，确保伪标签来源绝对的拓扑连通性。
#         """
#         N = h_idx.shape[0]
#         propagated_conf = torch.zeros(N, dtype=torch.float, device=self.device)
#         global_mean = base_edge_conf.mean() if base_edge_conf.numel() > 0 else torch.tensor(0.5, device=self.device)
        
#         # 游走权重超参数：偏好相同关系
#         same_rel_weight = 5.0 
#         diff_rel_weight = 1.0

#         for i in range(N):
#             h = h_idx[i].item()
#             t = t_idx[i].item()
#             r = r_idx[i].item()

#             # 寻找物理相连的 1-hop 邻居边
#             mask_h_neighbors = (base_edge_idx[0] == h) | (base_edge_idx[1] == h)
#             mask_t_neighbors = (base_edge_idx[0] == t) | (base_edge_idx[1] == t)
#             local_edges_mask = mask_h_neighbors | mask_t_neighbors

#             # 如果是游离孤岛，则退化为全局关系均值
#             if not local_edges_mask.any():
#                 same_rel_global_mask = (base_edge_type == r)
#                 if same_rel_global_mask.any():
#                     propagated_conf[i] = base_edge_conf[same_rel_global_mask].mean()
#                 else:
#                     propagated_conf[i] = global_mean
#                 continue

#             neighbor_rels = base_edge_type[local_edges_mask]
#             neighbor_confs = base_edge_conf[local_edges_mask]

#             # 分配游走转移权重
#             walk_weights = torch.where(
#                 neighbor_rels == r, 
#                 torch.tensor(same_rel_weight, dtype=torch.float, device=self.device), 
#                 torch.tensor(diff_rel_weight, dtype=torch.float, device=self.device)
#             )

#             total_weight = walk_weights.sum()
#             normalized_weights = walk_weights / (total_weight + 1e-6)
#             propagated_conf[i] = (normalized_weights * neighbor_confs).sum()

#         return propagated_conf

#     def _propagate_then_finetune(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, has_label_mask=None, known_conf=None):
#         """
#         阶段 1：两阶段课程微调 (立锚与扩散)。
#         在绝对隔离旧知识的前提下，先用真实标签把新实体“钉”在正确的特征空间，再放开伪标签做柔和扩散。
#         """
#         # 超参数读取
#         anchor_steps = getattr(self.args, 'anchor_steps', 3) if self.args else 3
#         diffuse_steps = getattr(self.args, 'finetune_steps', 5) if self.args else 5
#         mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01
#         lambda_ent_reg = getattr(self.args, 'lambda_ent_reg', 0.1) if self.args else 0.1 
#         dynamic_update_interval = getattr(self.args, 'dynamic_update_interval', 2) if self.args else 2
#         alpha_new = getattr(self.args, 'alpha_new_supervision', 0.3) if self.args else 0.3
#         alpha_labeled = getattr(self.args, 'alpha_labeled_supervision', 1.0) if self.args else 1.0
#         base_num_ent = self.dataset.base_num_ent

#         if has_label_mask is None:
#             has_label_mask = torch.zeros(h_idx.shape[0], dtype=torch.bool, device=self.device)
#         if known_conf is None:
#             known_conf = torch.zeros(h_idx.shape[0], dtype=torch.float, device=self.device)
        
#         unlabeled_mask = ~has_label_mask

#         # 步骤 1.0: 拓扑置信度传播初始化
#         with torch.no_grad():
#             propagated_conf = self._topological_confidence_propagation(
#                 h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf
#             )
#             if has_label_mask.any():
#                 propagated_conf[has_label_mask] = known_conf[has_label_mask]

#         # 合并出完整的输入子图
#         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
#         combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
#         combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
#         combined_edge_conf = torch.cat([base_edge_conf, propagated_conf], dim=0)

#         # 准备优化器，锁定底层旧有表征
#         old_mlp_params = {
#             name: param.detach().clone() for name, param in self.model.mlp_mean.named_parameters() if param.requires_grad
#         }
#         trainable_params = [self.model.entity_emb.weight] + [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
#         optimizer = optim.Adam(trainable_params, lr=self.lr)

#         n_ents_s1 = self.model.entity_emb.weight.shape[0]
#         stage1_update_mask = torch.zeros(n_ents_s1, dtype=torch.bool, device=self.device)
#         stage1_update_mask[base_num_ent:] = True
#         hook_s1 = self.model.entity_emb.weight.register_hook(
#             self._make_selective_grad_hook(stage1_update_mask)
#         )

#         self.model.train()
#         # 记录新实体的纯拓扑初始化中心，用于后续防过拟合
#         initial_new_ent_emb = self.model.entity_emb.weight[base_num_ent:].detach().clone()

#         # ==================================================================
#         # Sub-stage 1.1: 立锚 (Anchor)
#         # 仅使用有确凿标签的新事实进行强微调，防止伪标签和模型本身的不稳定性带来空间坍塌。
#         # ==================================================================
#         if has_label_mask.any() and anchor_steps > 0:
#             for step_a in range(anchor_steps):
#                 optimizer.zero_grad()
#                 z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

#                 mu_labeled_pred, _ = self.model.predict(
#                     z[h_idx[has_label_mask]], r_idx[has_label_mask], z[t_idx[has_label_mask]]
#                 )
                
#                 loss_labeled = alpha_labeled * F.mse_loss(mu_labeled_pred, known_conf[has_label_mask])

#                 # 实体空间防漂移正则：阻止新实体为了迎合某唯一标签而毁坏其图谱语义
#                 curr_new_ent_emb = self.model.entity_emb.weight[base_num_ent:]
#                 loss_ent_reg = lambda_ent_reg * torch.mean((curr_new_ent_emb - initial_new_ent_emb) ** 2)

#                 # MLP 功能性锚定
#                 loss_mlp_anchor = torch.tensor(0.0, device=self.device)
#                 for name, param in self.model.mlp_mean.named_parameters():
#                     if param.requires_grad and name in old_mlp_params:
#                         loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
#                 loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

#                 # # 🌟🌟🌟 在这里插入 Print 语句 🌟🌟🌟
#                 # print(f"[Stage 1.1] 真实标签Loss: {loss_labeled.item():.6f} | 实体空间正则: {loss_ent_reg.item():.6f} | MLP锚定正则: {loss_mlp_anchor.item():.6f}")

#                 loss_anchor_total = loss_labeled + loss_mlp_anchor + loss_ent_reg
#                 loss_anchor_total.backward()

#                 torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
#                 optimizer.step()

#                 # 🌟 新增：见好就收的早停机制
#                 if loss_labeled.item() < 0.015:
#                     print(f"--> [Stage 1.1] 真实标签Loss已达标 ({loss_labeled.item():.4f})，提前在第 {step_a} 步退出立锚，防止过拟合！")
#                     break

#         # 🌟 新增抢救逻辑：利用立锚后的模型，重新生成更高质量的无标签伪目标！
#         if unlabeled_mask.any():
#             self.model.eval()
#             with torch.no_grad():
#                 # 用刚被真实标签训练过的实体表征，重新预测一次无标签事实
#                 z_tmp = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
#                 mu_better_pseudo, _ = self.model.predict(z_tmp[h_idx], r_idx, z_tmp[t_idx])
                
#                 # 覆盖掉之前那个纯靠拓扑算出来的、可能错得离谱的 propagated_conf
#                 propagated_conf[unlabeled_mask] = mu_better_pseudo[unlabeled_mask].detach()
#             self.model.train()

#         # ==================================================================
#         # Sub-stage 1.2: 扩散 (Diffuse)
#         # 引入无标签事实共训，同时赋予有标签数据极高的统治权重，避免被带噪梯度反噬。
#         # ==================================================================
#         for step_i in range(diffuse_steps):
#             optimizer.zero_grad()
#             z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

#             mu_new_pred, _ = self.model.predict(z[h_idx], r_idx, z[t_idx])
#             ramp = min(1.0, (step_i + 1) / diffuse_steps)

#             if has_label_mask.any():
#                 # 权重放大 5 倍，确立真实标签霸权
#                 loss_labeled = (alpha_labeled * 5.0) * F.mse_loss(
#                     mu_new_pred[has_label_mask], known_conf[has_label_mask]
#                 )
#             else:
#                 loss_labeled = torch.tensor(0.0, device=self.device)

#             if unlabeled_mask.any():
#                 loss_new = alpha_new * ramp * F.mse_loss(
#                     mu_new_pred[unlabeled_mask], propagated_conf[unlabeled_mask].detach()
#                 )
#             else:
#                 loss_new = torch.tensor(0.0, device=self.device)

#             loss_mlp_anchor = torch.tensor(0.0, device=self.device)
#             for name, param in self.model.mlp_mean.named_parameters():
#                 if param.requires_grad and name in old_mlp_params:
#                     loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
#             loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

#             # # 🌟🌟🌟 在这里插入 Print 语句 🌟🌟🌟
#             # print(f"[Stage 1.2 - Step {step_i}] 真实标签Loss: {loss_labeled.item():.6f} | 伪标签Loss: {loss_new.item():.6f} | MLP锚定正则: {loss_mlp_anchor.item():.6f}")

#             loss_total = loss_labeled + loss_new + loss_mlp_anchor
#             loss_total.backward()
            
#             torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
#             optimizer.step()

#             # 动态刷新伪标签追踪
#             if dynamic_update_interval > 0 and step_i > 0 and step_i % dynamic_update_interval == 0:
#                 with torch.no_grad():
#                     new_confs = mu_new_pred.detach().clamp(0.0, 1.0)
#                     if has_label_mask.any():
#                         new_confs[has_label_mask] = known_conf[has_label_mask]
#                     combined_edge_conf = torch.cat([base_edge_conf, new_confs], dim=0)

#         hook_s1.remove()

#         # 输出最终微调完毕的预测值和方差
#         self.model.eval()
#         with torch.no_grad():
#             z_final = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
#             new_mu, new_sigma = self.model.predict(z_final[h_idx], r_idx, z_final[t_idx])

#         return new_mu.detach(), new_sigma.detach()

#     def _get_k_hop_subgraph(self, center_nodes, edge_index, num_hops=2):
#         """为中心节点周围的 K-hop 物理邻域快速生成布尔掩码。"""
#         visited_nodes = center_nodes.clone()
#         edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=self.device)

#         for _ in range(num_hops):
#             mask_h = torch.isin(edge_index[0], visited_nodes)
#             mask_t = torch.isin(edge_index[1], visited_nodes)
#             hop_edge_mask = mask_h | mask_t
            
#             edge_mask |= hop_edge_mask
            
#             if hop_edge_mask.any():
#                 visited_nodes = torch.unique(edge_index[:, edge_mask])
#             else:
#                 break

#         return edge_mask

#     def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq, has_label_mask=None):
#         """
#         评估新事实注入对旧图谱引发的局部因果干预效果 (ITE)。
#         极具扩展性：自动截取局部 K-hop 子图计算替代 O(E) 的全图低效计算。
#         """
#         self.model.eval()

#         new_ents = torch.cat([h_idx, t_idx]).unique()
#         num_hops = getattr(self.args, 'causal_num_hops', 2) if self.args else 2
#         sub_mask = self._get_k_hop_subgraph(new_ents, base_edge_idx, num_hops=num_hops)
        
#         E_total = base_edge_idx.shape[1]
#         S_tau_global = torch.zeros(E_total, dtype=torch.float, device=self.device)
#         mu_with_global = mu_without.clone()
#         sigma_sq_old_global = torch.zeros(E_total, dtype=torch.float, device=self.device)

#         if not sub_mask.any():
#             return S_tau_global, mu_with_global, sigma_sq_old_global

#         sub_base_edge_idx = base_edge_idx[:, sub_mask]
#         sub_base_edge_type = base_edge_type[sub_mask]
#         sub_base_edge_conf = base_edge_conf[sub_mask]
#         sub_mu_without = mu_without[sub_mask]

#         new_edge_index = torch.stack([h_idx, t_idx], dim=0)
#         combined_sub_edge_idx = torch.cat([sub_base_edge_idx, new_edge_index], dim=1)
#         combined_sub_edge_type = torch.cat([sub_base_edge_type, r_idx], dim=0)
#         combined_sub_edge_conf = torch.cat([sub_base_edge_conf, new_mu], dim=0)

#         with torch.no_grad():
#             curr_z_sub = self.model(combined_sub_edge_idx, combined_sub_edge_type, combined_sub_edge_conf)
            
#             sub_mu_with, sub_sigma_sq_old = self.model.predict(
#                 curr_z_sub[sub_base_edge_idx[0]], sub_base_edge_type, curr_z_sub[sub_base_edge_idx[1]]
#             )

#             # 基础个体干预效果 (ITE)
#             ITE_sub = torch.abs(sub_mu_with - sub_mu_without)

#             # 干预源可靠性：带有确定性标签的事实不确定性为零，拥有最大的传导话语权
#             per_fact_sigma = new_sigma_sq.clone()
#             if has_label_mask is not None and has_label_mask.any():
#                 per_fact_sigma[has_label_mask] = 0.0
#             intervention_reliability = 1.0 / (1.0 + per_fact_sigma.mean().item())

#             # 混杂因子节点度数惩罚：抑制超级枢纽节点传递的过度影响
#             t_nodes_sub = sub_base_edge_idx[1]
#             node_degrees = torch.bincount(t_nodes_sub, minlength=self.dataset.base_num_ent)
#             target_degrees = node_degrees[t_nodes_sub].float()
#             confounder_penalty = 1.0 / torch.log2(target_degrees + 2.0)

#             S_tau_sub = ITE_sub * intervention_reliability * confounder_penalty * self.gamma
#             S_tau_sub = torch.where(S_tau_sub < 1e-4, torch.zeros_like(S_tau_sub), S_tau_sub)

#             # 将 K-hop 子图的快速计算结果无缝拼回全局张量
#             S_tau_global[sub_mask] = S_tau_sub
#             mu_with_global[sub_mask] = sub_mu_with
#             sigma_sq_old_global[sub_mask] = sub_sigma_sq_old

#         return S_tau_global, mu_with_global, sigma_sq_old_global

#     def _bayesian_belief_filtering(self, base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau):
#         """融合因果干预强度与认知不确定性，在旧信念与新信念间实现平滑贝叶斯过渡。"""
#         epsilon = getattr(self.args, 'epsilon', 1e-4)

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
#         """
#         阶段 2：局部精炼与记忆保护。
#         仅对被深度牵连的局部旧知识解锁梯度，运用功能性输出锚定多重约束完成无缝融合。
#         """
#         threshold = getattr(self.args, 'influence_threshold', 0.01)
#         lambda_reg = min(getattr(self.args, 'lambda_reg', 0.001), 0.01)
#         func_anchor_ratio = getattr(self.args, 'func_anchor_ratio', 0.9)
#         alpha = 1.0
#         steps = getattr(self.args, 'refine_steps', 3)
#         base_num_ent = self.dataset.base_num_ent

#         affected_mask = torch.ones_like(S_tau, dtype=torch.bool) if self.ablation_mode == "wo_causal" else (S_tau > threshold)
#         real_affected_count = affected_mask.sum().item()

#         # 圈定本次受到干预并被激活参数更新的实体交集
#         new_ents = torch.cat([h_idx, t_idx]).unique()
#         if affected_mask.any():
#             affected_edges = base_edge_idx[:, affected_mask]
#             affected_old_ents = torch.cat([affected_edges[0], affected_edges[1]]).unique()
#         else:
#             affected_old_ents = torch.empty(0, dtype=torch.long, device=self.device)

#         ents_to_update = torch.cat([new_ents, affected_old_ents]).unique()
#         old_ents_for_reg = affected_old_ents[affected_old_ents < base_num_ent] if affected_old_ents.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)

#         update_mask = torch.zeros(self.model.entity_emb.weight.shape[0], dtype=torch.bool, device=self.device)
#         update_mask[ents_to_update] = True

#         # 生成功能锚定所需的完美预测基准靶标
#         with torch.no_grad():
#             if affected_mask.any():
#                 affected_edges_snap = base_edge_idx[:, affected_mask]
#                 affected_rels_snap = base_edge_type[affected_mask]
#                 mu_old_affected, _ = self.model.predict(
#                     old_z[affected_edges_snap[0]], affected_rels_snap, old_z[affected_edges_snap[1]]
#                 )
#                 mu_old_affected = mu_old_affected.detach()
#             else:
#                 mu_old_affected = None

#         self.model.train()
#         optimizer = optim.Adam([self.model.entity_emb.weight], lr=self.lr)
#         hook_s2 = self.model.entity_emb.weight.register_hook(self._make_selective_grad_hook(update_mask))

#         for _ in range(steps):
#             optimizer.zero_grad()
#             curr_z = self.model.entity_emb.weight

#             # 子损失 1：护航当前核心，确保新事实的最终打分
#             mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
#             loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)

#             # 子损失 2：要求受波及的旧事实逼近平滑后的贝叶斯新信念 c_new
#             if affected_mask.any():
#                 affected_edges = base_edge_idx[:, affected_mask]
#                 affected_rels = base_edge_type[affected_mask]
#                 mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]])
#                 loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
#             else:
#                 loss_affected = torch.tensor(0.0, device=self.device)

#             # 子损失 3：受牵连旧实体的基底 L2 表征正则约束
#             if old_ents_for_reg.numel() > 0:
#                 loss_reg_old = lambda_reg * torch.mean((curr_z[old_ents_for_reg] - old_z[old_ents_for_reg].detach()) ** 2)
#             else:
#                 loss_reg_old = torch.tensor(0.0, device=self.device)

#             # 子损失 4：功能性锚定，防止局部知识链由于单边改动而连锁崩溃
#             if affected_mask.any() and mu_old_affected is not None:
#                 affected_edges_cur = base_edge_idx[:, affected_mask]
#                 affected_rels_cur = base_edge_type[affected_mask]
#                 mu_cur_affected, _ = self.model.predict(
#                     curr_z[affected_edges_cur[0]], affected_rels_cur, curr_z[affected_edges_cur[1]]
#                 )
#                 func_loss = F.mse_loss(mu_cur_affected, mu_old_affected)
#                 loss_reg = lambda_reg * func_anchor_ratio * func_loss + loss_reg_old
#             else:
#                 loss_reg = loss_reg_old

#             loss_total = loss_new + alpha * loss_affected + loss_reg
#             loss_total.backward()
#             optimizer.step()

#         hook_s2.remove()
#         return real_affected_count

#     def _update_dataset_belief(self, base_edge_idx, base_edge_type, updated_beliefs):
#         """将 GPU 中计算完毕的最新张量级置信状态下放同步回底层的字典结构。"""
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
    """
    动态更新知识图谱中事实的置信度得分。
    
    该更新器通过结合拓扑标签传播、课程微调、K-hop因果影响评估和局部的贝叶斯表征精炼等流水线，
    来稳健地处理新事实（以及知识库外/OOKB实体）的插入，并极力避免灾难性遗忘。
    """
    
    def __init__(self, model, dataset, lr=0.001, gamma=0.8, device="cpu", args=None, ablation_mode="full"):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.args = args

        # 统一处理 ablation_mode 为列表，方便进行多重/组合消融
        if isinstance(ablation_mode, str):
            if ablation_mode.strip() == "" or ablation_mode.lower() == "full":
                self.ablation_mode = []
            else:
                self.ablation_mode = [m.strip() for m in ablation_mode.split(",")]
        elif isinstance(ablation_mode, list):
            self.ablation_mode = ablation_mode
        else:
            self.ablation_mode = []

        # 冻结 MLP 和关系嵌入以保留全局图结构
        for param in self.model.mlp_mean.parameters():
            param.requires_grad = False
        for param in self.model.mlp_var.parameters():
            param.requires_grad = False
        self.model.relation_emb.weight.requires_grad = False

        # 全部解冻 mlp_mean，交由 mlp_anchor_coeff 去约束
        for name, param in self.model.mlp_mean.named_parameters():
            param.requires_grad = False 

    def step(self, new_facts_batch):
        """
        执行单步更新，处理一个批次的增量新事实。
        """
        # 解析输入数据
        h_idx = torch.tensor([f[0] for f in new_facts_batch], dtype=torch.long, device=self.device)
        r_idx = torch.tensor([f[1] for f in new_facts_batch], dtype=torch.long, device=self.device)
        t_idx = torch.tensor([f[2] for f in new_facts_batch], dtype=torch.long, device=self.device)

        has_label_list = [len(f) > 3 and f[3] is not None for f in new_facts_batch]
        has_label_mask = torch.tensor(has_label_list, dtype=torch.bool, device=self.device)
        known_conf_list = [float(f[3]) if (len(f) > 3 and f[3] is not None) else 0.0 for f in new_facts_batch]
        known_conf = torch.tensor(known_conf_list, dtype=torch.float, device=self.device)

        # 提取基础图谱数据
        base_edge_idx, base_edge_type, base_edge_conf = self.dataset.get_base_graph_data()
        base_edge_idx = base_edge_idx.to(self.device)
        base_edge_type = base_edge_type.to(self.device)
        base_edge_conf = base_edge_conf.to(self.device)

        # 阶段 0: 针对纯新实体的安全几何初始化
        self._init_new_entities(h_idx, r_idx, t_idx)

        # 快照保存基础模型的预测输出和原始嵌入
        self.model.eval()
        with torch.no_grad():
            old_z = self.model(base_edge_idx, base_edge_type, base_edge_conf)
            mu_without, _ = self.model.predict(old_z[base_edge_idx[0]], base_edge_type, old_z[base_edge_idx[1]])
            old_raw_emb = self.model.entity_emb.weight.detach().clone()

        # 阶段 1: 两阶段防过拟合课程微调 (立锚与扩散)
        raw_new_mu, new_sigma_sq = self._propagate_then_finetune(
            h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, has_label_mask, known_conf
        )
        
        # 直接使用原始预测值往下走
        new_mu = raw_new_mu

        # 阶段 2: 基于 K-hop 子图的因果推断评估
        S_tau, mu_with, sigma_sq_old = self._compute_causal_influence(
            base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq, has_label_mask
        )

        # 阶段 3: 贝叶斯信念滤波
        c_new, c_old = self._bayesian_belief_filtering(
            base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau
        )

        # 阶段 4: 局部表征精炼
        real_affected_count = self._local_representation_refinement(
            base_edge_idx, base_edge_type, c_new, old_raw_emb, S_tau, h_idx, r_idx, t_idx, new_mu
        )

        # 更新数据集的全局状态
        self._update_dataset_belief(base_edge_idx, base_edge_type, c_new)

        # 将新事实的置信度写回字典
        new_mu_np = new_mu.detach().cpu().numpy()
        known_conf_np = known_conf.cpu().numpy()
        has_label_np = has_label_mask.cpu().numpy()
        h_np = h_idx.cpu().numpy()
        r_np = r_idx.cpu().numpy()
        t_np = t_idx.cpu().numpy()
        
        for i in range(len(h_np)):
            fact_tuple = (int(h_np[i]), int(r_np[i]), int(t_np[i]))
            self.dataset.belief_state[fact_tuple] = float(known_conf_np[i]) if has_label_np[i] else float(new_mu_np[i])

        change_tensor = torch.abs(c_new - c_old)
        return new_mu.detach(), change_tensor.mean().item(), change_tensor.max().item(), real_affected_count

    def _init_new_entities(self, h_idx, r_idx, t_idx):
        """
        使用几何均值初始化知识库外（OOKB）新实体的嵌入表征。
        """
        with torch.no_grad():
            ent_weight = self.model.entity_emb.weight
            new_ents = self.dataset.new_entities

            if not new_ents:
                return

            # Ablation 1: 移除几何中心初始化，退化为纯随机初始化
            if "wo_geom_init" in self.ablation_mode:
                for ent_id in new_ents:
                    torch.nn.init.normal_(ent_weight[ent_id].unsqueeze(0), mean=0.0, std=0.1)
                return

            new_ents_tensor = torch.tensor(list(new_ents), dtype=torch.long, device=self.device)

            for ent_id in new_ents:
                msgs = []
                
                # 当新实体作为头实体时
                mask_h = (h_idx == ent_id)
                if mask_h.any():
                    t_neighbors = t_idx[mask_h]
                    old_mask = ~torch.isin(t_neighbors, new_ents_tensor)
                    if old_mask.any():
                        msgs.append(ent_weight[t_neighbors[old_mask]].mean(dim=0))

                # 当新实体作为尾实体时
                mask_t = (t_idx == ent_id)
                if mask_t.any():
                    h_neighbors = h_idx[mask_t]
                    old_mask = ~torch.isin(h_neighbors, new_ents_tensor)
                    if old_mask.any():
                        msgs.append(ent_weight[h_neighbors[old_mask]].mean(dim=0))

                # 聚合表征
                if msgs:
                    ent_weight[ent_id] = torch.stack(msgs).mean(dim=0)
                else:
                    neighbors = []
                    if mask_h.any():
                        neighbors.extend(t_idx[mask_h].tolist())
                    if mask_t.any():
                        neighbors.extend(h_idx[mask_t].tolist())
                    if neighbors:
                        neighbor_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
                        ent_weight[ent_id] = ent_weight[neighbor_tensor].mean(dim=0)

    def _relation_aware_calibration(self, new_mu, new_sigma_sq, r_idx, base_edge_type, base_edge_conf, has_label_mask=None, known_conf=None):
        """(当前逻辑未调用) 利用关系级别的贝叶斯先验来校准模型对新事实的预测结果。"""
        calibrated = torch.zeros_like(new_mu)
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
            if has_label_mask is not None and has_label_mask[i]:
                calibrated[i] = known_conf[i]
                continue
            r = r_idx[i].item()
            stats = r_stats.get(r, {'mean': global_mean, 'var': global_var, 'count': 1})
            prior_mu = stats['mean']
            prior_precision = 1.0 / (stats['var'] + 1e-6)
            model_precision = 1.0 / (new_sigma_sq[i].item() + 1e-6)
            w = model_precision / (model_precision + prior_precision)
            calibrated[i] = w * new_mu[i] + (1 - w) * prior_mu
        return calibrated

    @staticmethod
    def _make_selective_grad_hook(update_mask):
        def hook(grad):
            grad_clone = grad.clone()
            grad_clone[~update_mask] = 0.0
            return grad_clone
        return hook

    def _topological_confidence_propagation(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf):
        """利用 1-hop 图拓扑随机游走，将先验置信度传播给无标签的新事实。"""
        N = h_idx.shape[0]
        propagated_conf = torch.zeros(N, dtype=torch.float, device=self.device)
        global_mean = base_edge_conf.mean() if base_edge_conf.numel() > 0 else torch.tensor(0.5, device=self.device)
        
        same_rel_weight = 5.0 
        diff_rel_weight = 1.0

        for i in range(N):
            h = h_idx[i].item()
            t = t_idx[i].item()
            r = r_idx[i].item()

            mask_h_neighbors = (base_edge_idx[0] == h) | (base_edge_idx[1] == h)
            mask_t_neighbors = (base_edge_idx[0] == t) | (base_edge_idx[1] == t)
            local_edges_mask = mask_h_neighbors | mask_t_neighbors

            if not local_edges_mask.any():
                same_rel_global_mask = (base_edge_type == r)
                if same_rel_global_mask.any():
                    propagated_conf[i] = base_edge_conf[same_rel_global_mask].mean()
                else:
                    propagated_conf[i] = global_mean
                continue

            neighbor_rels = base_edge_type[local_edges_mask]
            neighbor_confs = base_edge_conf[local_edges_mask]

            walk_weights = torch.where(
                neighbor_rels == r, 
                torch.tensor(same_rel_weight, dtype=torch.float, device=self.device), 
                torch.tensor(diff_rel_weight, dtype=torch.float, device=self.device)
            )

            total_weight = walk_weights.sum()
            normalized_weights = walk_weights / (total_weight + 1e-6)
            propagated_conf[i] = (normalized_weights * neighbor_confs).sum()

        return propagated_conf

    def _propagate_then_finetune(self, h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf, has_label_mask=None, known_conf=None):
        """阶段 1：两阶段课程微调 (立锚与扩散)。"""
        anchor_steps = getattr(self.args, 'anchor_steps', 3) if self.args else 3
        diffuse_steps = getattr(self.args, 'finetune_steps', 5) if self.args else 5
        mlp_anchor_coeff = getattr(self.args, 'mlp_anchor_coeff', 0.01) if self.args else 0.01
        lambda_ent_reg = getattr(self.args, 'lambda_ent_reg', 0.1) if self.args else 0.1 
        dynamic_update_interval = getattr(self.args, 'dynamic_update_interval', 2) if self.args else 2
        alpha_new = getattr(self.args, 'alpha_new_supervision', 0.3) if self.args else 0.3
        alpha_labeled = getattr(self.args, 'alpha_labeled_supervision', 1.0) if self.args else 1.0
        base_num_ent = self.dataset.base_num_ent

        # Ablation 3: 强行跳过立锚阶段
        if "wo_anchor" in self.ablation_mode:
            anchor_steps = 0
            
        # Ablation 4: 强行干掉实体空间防漂移正则
        if "wo_ent_reg" in self.ablation_mode:
            lambda_ent_reg = 0.0

        if has_label_mask is None:
            has_label_mask = torch.zeros(h_idx.shape[0], dtype=torch.bool, device=self.device)
        if known_conf is None:
            known_conf = torch.zeros(h_idx.shape[0], dtype=torch.float, device=self.device)
        
        unlabeled_mask = ~has_label_mask

        # 步骤 1.0: 拓扑置信度传播初始化
        with torch.no_grad():
            propagated_conf = self._topological_confidence_propagation(
                h_idx, r_idx, t_idx, base_edge_idx, base_edge_type, base_edge_conf
            )
            
            # Ablation 2: 移除拓扑传播，强行退化为全局均值
            if "wo_topo_prop" in self.ablation_mode:
                global_mean = base_edge_conf.mean() if base_edge_conf.numel() > 0 else 0.5
                propagated_conf = torch.full_like(propagated_conf, global_mean)

            if has_label_mask.any():
                propagated_conf[has_label_mask] = known_conf[has_label_mask]

        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_edge_index = torch.cat([base_edge_idx, new_edge_index], dim=1)
        combined_edge_type = torch.cat([base_edge_type, r_idx], dim=0)
        combined_edge_conf = torch.cat([base_edge_conf, propagated_conf], dim=0)

        old_mlp_params = {
            name: param.detach().clone() for name, param in self.model.mlp_mean.named_parameters() if param.requires_grad
        }
        trainable_params = [self.model.entity_emb.weight] + [p for p in self.model.mlp_mean.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr)

        n_ents_s1 = self.model.entity_emb.weight.shape[0]
        stage1_update_mask = torch.zeros(n_ents_s1, dtype=torch.bool, device=self.device)
        stage1_update_mask[base_num_ent:] = True
        hook_s1 = self.model.entity_emb.weight.register_hook(
            self._make_selective_grad_hook(stage1_update_mask)
        )

        self.model.train()
        initial_new_ent_emb = self.model.entity_emb.weight[base_num_ent:].detach().clone()

        # Sub-stage 1.1: 立锚 (Anchor)
        if has_label_mask.any() and anchor_steps > 0:
            for step_a in range(anchor_steps):
                optimizer.zero_grad()
                z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

                mu_labeled_pred, _ = self.model.predict(
                    z[h_idx[has_label_mask]], r_idx[has_label_mask], z[t_idx[has_label_mask]]
                )
                
                loss_labeled = alpha_labeled * F.mse_loss(mu_labeled_pred, known_conf[has_label_mask])

                curr_new_ent_emb = self.model.entity_emb.weight[base_num_ent:]
                loss_ent_reg = lambda_ent_reg * torch.mean((curr_new_ent_emb - initial_new_ent_emb) ** 2)

                loss_mlp_anchor = torch.tensor(0.0, device=self.device)
                for name, param in self.model.mlp_mean.named_parameters():
                    if param.requires_grad and name in old_mlp_params:
                        loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
                loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

                loss_anchor_total = loss_labeled + loss_mlp_anchor + loss_ent_reg
                loss_anchor_total.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                if loss_labeled.item() < 0.015:
                    break

        if unlabeled_mask.any() and ("wo_anchor" not in self.ablation_mode):
            self.model.eval()
            with torch.no_grad():
                z_tmp = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
                mu_better_pseudo, _ = self.model.predict(z_tmp[h_idx], r_idx, z_tmp[t_idx])
                propagated_conf[unlabeled_mask] = mu_better_pseudo[unlabeled_mask].detach()
            self.model.train()

        # Sub-stage 1.2: 扩散 (Diffuse)
        for step_i in range(diffuse_steps):
            optimizer.zero_grad()
            z = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)

            mu_new_pred, _ = self.model.predict(z[h_idx], r_idx, z[t_idx])
            ramp = min(1.0, (step_i + 1) / diffuse_steps)

            if has_label_mask.any():
                loss_labeled = (alpha_labeled * 5.0) * F.mse_loss(
                    mu_new_pred[has_label_mask], known_conf[has_label_mask]
                )
            else:
                loss_labeled = torch.tensor(0.0, device=self.device)

            if unlabeled_mask.any():
                loss_new = alpha_new * ramp * F.mse_loss(
                    mu_new_pred[unlabeled_mask], propagated_conf[unlabeled_mask].detach()
                )
            else:
                loss_new = torch.tensor(0.0, device=self.device)

            loss_mlp_anchor = torch.tensor(0.0, device=self.device)
            for name, param in self.model.mlp_mean.named_parameters():
                if param.requires_grad and name in old_mlp_params:
                    loss_mlp_anchor = loss_mlp_anchor + torch.mean((param - old_mlp_params[name]) ** 2)
            loss_mlp_anchor = mlp_anchor_coeff * loss_mlp_anchor

            loss_total = loss_labeled + loss_new + loss_mlp_anchor
            loss_total.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            if dynamic_update_interval > 0 and step_i > 0 and step_i % dynamic_update_interval == 0:
                with torch.no_grad():
                    new_confs = mu_new_pred.detach().clamp(0.0, 1.0)
                    if has_label_mask.any():
                        new_confs[has_label_mask] = known_conf[has_label_mask]
                    combined_edge_conf = torch.cat([base_edge_conf, new_confs], dim=0)

        hook_s1.remove()

        self.model.eval()
        with torch.no_grad():
            z_final = self.model(combined_edge_index, combined_edge_type, combined_edge_conf)
            new_mu, new_sigma = self.model.predict(z_final[h_idx], r_idx, z_final[t_idx])

        return new_mu.detach(), new_sigma.detach()

    def _get_k_hop_subgraph(self, center_nodes, edge_index, num_hops=2):
        visited_nodes = center_nodes.clone()
        edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=self.device)

        for _ in range(num_hops):
            mask_h = torch.isin(edge_index[0], visited_nodes)
            mask_t = torch.isin(edge_index[1], visited_nodes)
            hop_edge_mask = mask_h | mask_t
            
            edge_mask |= hop_edge_mask
            
            if hop_edge_mask.any():
                visited_nodes = torch.unique(edge_index[:, edge_mask])
            else:
                break

        return edge_mask

    def _compute_causal_influence(self, base_edge_idx, base_edge_type, base_edge_conf, mu_without, h_idx, r_idx, t_idx, new_mu, new_sigma_sq, has_label_mask=None):
        self.model.eval()

        new_ents = torch.cat([h_idx, t_idx]).unique()
        num_hops = getattr(self.args, 'causal_num_hops', 2) if self.args else 2
        sub_mask = self._get_k_hop_subgraph(new_ents, base_edge_idx, num_hops=num_hops)
        
        E_total = base_edge_idx.shape[1]
        S_tau_global = torch.zeros(E_total, dtype=torch.float, device=self.device)
        mu_with_global = mu_without.clone()
        sigma_sq_old_global = torch.zeros(E_total, dtype=torch.float, device=self.device)

        if not sub_mask.any():
            return S_tau_global, mu_with_global, sigma_sq_old_global

        sub_base_edge_idx = base_edge_idx[:, sub_mask]
        sub_base_edge_type = base_edge_type[sub_mask]
        sub_base_edge_conf = base_edge_conf[sub_mask]
        sub_mu_without = mu_without[sub_mask]

        new_edge_index = torch.stack([h_idx, t_idx], dim=0)
        combined_sub_edge_idx = torch.cat([sub_base_edge_idx, new_edge_index], dim=1)
        combined_sub_edge_type = torch.cat([sub_base_edge_type, r_idx], dim=0)
        combined_sub_edge_conf = torch.cat([sub_base_edge_conf, new_mu], dim=0)

        with torch.no_grad():
            curr_z_sub = self.model(combined_sub_edge_idx, combined_sub_edge_type, combined_sub_edge_conf)
            
            sub_mu_with, sub_sigma_sq_old = self.model.predict(
                curr_z_sub[sub_base_edge_idx[0]], sub_base_edge_type, curr_z_sub[sub_base_edge_idx[1]]
            )

            ITE_sub = torch.abs(sub_mu_with - sub_mu_without)

            per_fact_sigma = new_sigma_sq.clone()
            if has_label_mask is not None and has_label_mask.any():
                per_fact_sigma[has_label_mask] = 0.0
            intervention_reliability = 1.0 / (1.0 + per_fact_sigma.mean().item())

            t_nodes_sub = sub_base_edge_idx[1]
            node_degrees = torch.bincount(t_nodes_sub, minlength=self.dataset.base_num_ent)
            target_degrees = node_degrees[t_nodes_sub].float()
            confounder_penalty = 1.0 / torch.log2(target_degrees + 2.0)

            S_tau_sub = ITE_sub * intervention_reliability * confounder_penalty * self.gamma
            S_tau_sub = torch.where(S_tau_sub < 1e-4, torch.zeros_like(S_tau_sub), S_tau_sub)

            S_tau_global[sub_mask] = S_tau_sub
            mu_with_global[sub_mask] = sub_mu_with
            sigma_sq_old_global[sub_mask] = sub_sigma_sq_old

        return S_tau_global, mu_with_global, sigma_sq_old_global

    def _bayesian_belief_filtering(self, base_edge_idx, base_edge_type, mu_with, sigma_sq_old, S_tau):
        epsilon = getattr(self.args, 'epsilon', 1e-4)

        c_old = []
        h_list = base_edge_idx[0].cpu().numpy()
        t_list = base_edge_idx[1].cpu().numpy()
        r_list = base_edge_type.cpu().numpy()

        for i in range(len(h_list)):
            fact_tuple = (h_list[i], r_list[i], t_list[i])
            c_old.append(self.dataset.belief_state.get(fact_tuple, 0.5))

        c_old = torch.tensor(c_old, dtype=torch.float, device=self.device)

        # Ablation 6: 移除贝叶斯滤波，退化为算术均值
        if "wo_bayes" in self.ablation_mode:
            c_new = 0.5 * c_old + 0.5 * mu_with
        else:
            w_tau = S_tau / (S_tau + sigma_sq_old + epsilon)
            c_new = (1 - w_tau) * c_old + w_tau * mu_with

        return torch.clamp(c_new, 0.0, 1.0), c_old

    def _local_representation_refinement(self, base_edge_idx, base_edge_type, c_new, old_z, S_tau, h_idx, r_idx, t_idx, new_mu):
        threshold = getattr(self.args, 'influence_threshold', 0.01)
        lambda_reg = min(getattr(self.args, 'lambda_reg', 0.001), 0.01)
        func_anchor_ratio = getattr(self.args, 'func_anchor_ratio', 0.9)
        alpha = 1.0
        steps = getattr(self.args, 'refine_steps', 3)
        base_num_ent = self.dataset.base_num_ent

        # Ablation 7: 切断功能性输出锚定权重
        if "wo_func_anchor" in self.ablation_mode:
            func_anchor_ratio = 0.0

        # Ablation 5: 移除局部隔离，默认全部牵连
        if "wo_causal" in self.ablation_mode:
            affected_mask = torch.ones_like(S_tau, dtype=torch.bool)
        else:
            affected_mask = (S_tau > threshold)
            
        real_affected_count = affected_mask.sum().item()

        new_ents = torch.cat([h_idx, t_idx]).unique()
        if affected_mask.any():
            affected_edges = base_edge_idx[:, affected_mask]
            affected_old_ents = torch.cat([affected_edges[0], affected_edges[1]]).unique()
        else:
            affected_old_ents = torch.empty(0, dtype=torch.long, device=self.device)

        ents_to_update = torch.cat([new_ents, affected_old_ents]).unique()
        old_ents_for_reg = affected_old_ents[affected_old_ents < base_num_ent] if affected_old_ents.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)

        update_mask = torch.zeros(self.model.entity_emb.weight.shape[0], dtype=torch.bool, device=self.device)
        update_mask[ents_to_update] = True

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
        hook_s2 = self.model.entity_emb.weight.register_hook(self._make_selective_grad_hook(update_mask))

        for _ in range(steps):
            optimizer.zero_grad()
            curr_z = self.model.entity_emb.weight

            mu_pred_new, _ = self.model.predict(curr_z[h_idx], r_idx, curr_z[t_idx])
            loss_new = torch.mean((new_mu.detach() - mu_pred_new) ** 2)

            if affected_mask.any():
                affected_edges = base_edge_idx[:, affected_mask]
                affected_rels = base_edge_type[affected_mask]
                mu_pred_old, _ = self.model.predict(curr_z[affected_edges[0]], affected_rels, curr_z[affected_edges[1]])
                loss_affected = torch.mean((c_new[affected_mask] - mu_pred_old) ** 2)
            else:
                loss_affected = torch.tensor(0.0, device=self.device)

            if old_ents_for_reg.numel() > 0:
                loss_reg_old = lambda_reg * torch.mean((curr_z[old_ents_for_reg] - old_z[old_ents_for_reg].detach()) ** 2)
            else:
                loss_reg_old = torch.tensor(0.0, device=self.device)

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

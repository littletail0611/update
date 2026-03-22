# Base 模型模块：置信度感知 GNN + 异方差预测双分支 (均值 & 方差)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class ConfidenceAwareGNNLayer(MessagePassing):
    def __init__(self, emb_dim):
        super(ConfidenceAwareGNNLayer, self).__init__(aggr='add')
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_0 = nn.Linear(emb_dim, emb_dim)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, 1)
        )
        
        # 【新增】：置信度门控函数 g(c)
        # 这里使用一个单层可学习的 Sigmoid 映射，让模型自适应调整置信度缩放
        self.conf_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_type, edge_conf, rel_emb):
        r_features = rel_emb[edge_type]
        out = self.propagate(edge_index, x=x, r_features=r_features, edge_conf=edge_conf)
        out = out + self.W_0(x)
        return F.relu(out)

    def message(self, x_i, x_j, r_features, edge_conf, index):
        # 1. 计算特征驱动的 logit (e_ij)
        attn_input = torch.cat([x_j, r_features, x_i], dim=-1)
        e_ur = self.attn_mlp(attn_input).squeeze(-1)
        
        # 2. α_ij = softmax(e_ij)
        # 此时的 attention 完全由节点和边的语义特征决定，不受 edge_conf 干扰
        alpha = softmax(e_ur, index)
        
        # Message Dropout (仅在 model.train() 时生效)
        alpha = F.dropout(alpha, p=0.2, training=self.training)
        
        # 3. φ(x_j, r_ij)：基础消息内容
        base_msg = self.W_v(x_j) * r_features
        
        # 4. g(c_ij)：计算置信度门控强度
        # 将 edge_conf 从 [E] 扩展为 [E, 1] 输入到网络或参与广播
        gate = self.conf_gate(edge_conf.unsqueeze(-1))
        
        # 注意：如果你想用纯 Identity (即 g(c) = c)，请注释掉上面的 self.conf_gate，
        # 直接使用下面的代码即可：
        # gate = edge_conf.unsqueeze(-1)
        
        # 5. m_ij = α_ij * (g(c_ij) * φ(x_j, r_ij))
        msg = alpha.unsqueeze(-1) * (gate * base_msg)
        
        return msg


class HeteroscedasticBaseModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, num_layers=2, dropout_rate=0.3):
        super(HeteroscedasticBaseModel, self).__init__()
        self.emb_dim = emb_dim
        
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        
        self.gnn_layers = nn.ModuleList([
            ConfidenceAwareGNNLayer(emb_dim) for _ in range(num_layers)
        ])
        
        # 加入 Dropout 防止死记硬背
        self.mlp_mean = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()  
        )
        
        self.mlp_var = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, 1),
            nn.Softplus() 
        )

    def forward(self, edge_index, edge_type, edge_conf):
        x = self.entity_emb.weight
        rel_weights = self.relation_emb.weight
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_type, edge_conf, rel_weights)
        return x

    def predict(self, z_h, r_id, z_t):
        r_features = self.relation_emb(r_id) if r_id is None or r_id.dim() > 0 else self.relation_emb(torch.tensor([r_id], device=z_h.device))
        h_r_t_concat = torch.cat([z_h, r_features, z_t], dim=-1)
        
        mu = self.mlp_mean(h_r_t_concat).squeeze(-1)
        sigma_sq = self.mlp_var(h_r_t_concat).squeeze(-1)
        return mu, sigma_sq

    def heteroscedastic_loss(self, mu, sigma_sq, target_conf):
        eps = 1e-6
        mse_term = ((target_conf - mu) ** 2) / (2 * (sigma_sq + eps))
        reg_term = 0.5 * torch.log(sigma_sq + eps)
        return torch.mean(mse_term + reg_term)

    def continuous_contrastive_loss(self, z_h, r_id, z_t, labels, temperature=0.07):
        """优化版：带有硬性截断和更低温度系数的对比损失"""
        r_features = self.relation_emb(r_id)
        z_hrt = z_h + r_features + z_t 
        
        z_hrt = F.normalize(z_hrt, p=2, dim=1)
        feat_sim = torch.matmul(z_hrt, z_hrt.T) / temperature
        
        labels = labels.unsqueeze(1)
        label_diff = torch.abs(labels - labels.T)
        
        # 【优化】：使用更尖锐的高斯核带宽，并引入阈值截断
        # 只有置信度差异小于 0.2 的才被视为正样本(互相拉近)
        label_weight = torch.exp(- (label_diff ** 2) / 0.05)
        label_weight[label_diff > 0.2] = 0.0  # 硬性推远差异大的样本
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        label_weight.masked_fill_(mask, 0.0)
        
        num = torch.sum(label_weight * torch.exp(feat_sim), dim=1)
        den = torch.sum(torch.exp(feat_sim) * (~mask), dim=1)
        
        # 过滤掉 num 为 0 的异常样本，防止 log(0)
        valid_mask = num > 1e-6
        if not valid_mask.any():
            return torch.tensor(0.0, device=labels.device, requires_grad=True)
            
        loss_cl = -torch.log(num[valid_mask] / (den[valid_mask] + 1e-8)).mean()
        return loss_cl
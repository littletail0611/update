# baselines.py
# 7个持续学习基线模型，适配 UKG 增量置信度预测任务
# 每个基线共享 HeteroscedasticBaseModel 主干网络，实现相同的 .step() 接口

import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quadprog


# ─────────────────────────────────────────────────────────────────────────────
# 共用工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _init_new_entities(model, dataset, device):
    """对 OOKB 新实体用已见邻居的均值初始化 embedding，若无邻居则用随机噪声。"""
    if not dataset.new_entities:
        return
    with torch.no_grad():
        base_mean = model.entity_emb.weight[:dataset.base_num_ent].mean(0)
        for eid in dataset.new_entities:
            if eid < model.entity_emb.weight.shape[0]:
                model.entity_emb.weight[eid].copy_(
                    base_mean + 0.01 * torch.randn_like(base_mean)
                )


def _fact_conf(fact, belief_state):
    """返回事实的置信度：有标注用自身值，无标注从 belief_state 查找，否则用 0.5。"""
    if fact[3] is not None:
        return fact[3]
    return belief_state.get((fact[0], fact[1], fact[2]), 0.5)


def _build_combined_graph(dataset, device):
    """返回 Base + Inc_train 的合并图张量，供 GNN 前向推断使用。"""
    base_facts = dataset.base_train
    inc_facts = dataset.inc_train

    all_facts = base_facts + inc_facts
    if not all_facts:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float, device=device),
        )

    heads = torch.tensor([f[0] for f in all_facts], dtype=torch.long, device=device)
    tails = torch.tensor([f[2] for f in all_facts], dtype=torch.long, device=device)
    rels  = torch.tensor([f[1] for f in all_facts], dtype=torch.long, device=device)
    confs = torch.tensor(
        [_fact_conf(f, dataset.belief_state) for f in all_facts],
        dtype=torch.float, device=device,
    )
    return torch.stack([heads, tails]), rels, confs


def _base_graph_tensors(dataset, device):
    """仅返回 Base 图张量（用于记忆回放等需要干净 Base 图的场景）。"""
    edge_index, edge_type, edge_conf = dataset.get_base_graph_data()
    return edge_index.to(device), edge_type.to(device), edge_conf.to(device)


def _labeled_facts(facts):
    """过滤出有标注置信度（conf 不为 None）的事实。"""
    return [f for f in facts if f[3] is not None]


def _step_summary(model, dataset, new_facts, device):
    """
    计算 step() 的标准返回值:
      (new_mu, change_mean, change_max, affected_count)
    """
    ei, et, ec = _build_combined_graph(dataset, device)
    model.eval()
    with torch.no_grad():
        z = model(ei, et, ec)

    # 新事实的预测均值
    labeled = _labeled_facts(new_facts)
    if labeled:
        h_t = torch.tensor([f[0] for f in labeled], dtype=torch.long, device=device)
        r_t = torch.tensor([f[1] for f in labeled], dtype=torch.long, device=device)
        t_t = torch.tensor([f[2] for f in labeled], dtype=torch.long, device=device)
        mu_new, _ = model.predict(z[h_t], r_t, z[t_t])
    else:
        mu_new = torch.tensor([0.5], device=device)

    # 计算受影响的旧事实变化量（与 belief_state 对比）
    changes = []
    for (h, r, t), old_conf in dataset.belief_state.items():
        if h < z.shape[0] and t < z.shape[0]:
            r_tensor = torch.tensor([r], dtype=torch.long, device=device)
            with torch.no_grad():
                mu, _ = model.predict(z[[h]], r_tensor, z[[t]])
            delta = abs(mu.item() - old_conf)
            changes.append(delta)

    change_mean = float(np.mean(changes)) if changes else 0.0
    change_max  = float(np.max(changes))  if changes else 0.0
    affected    = int(sum(1 for d in changes if d > 0.005))

    return mu_new, change_mean, change_max, affected


def _update_belief_state(model, dataset, new_facts, device):
    """用模型当前预测更新 dataset.belief_state（新事实 + 旧事实）。"""
    ei, et, ec = _build_combined_graph(dataset, device)
    model.eval()
    with torch.no_grad():
        z = model(ei, et, ec)
        for f in new_facts:
            h, r, t = f[0], f[1], f[2]
            r_tensor = torch.tensor([r], dtype=torch.long, device=device)
            mu, _ = model.predict(z[[h]], r_tensor, z[[t]])
            dataset.belief_state[(h, r, t)] = mu.item()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CWR (CopyWeights with Re-init)
# ─────────────────────────────────────────────────────────────────────────────

class CWRUpdater:
    """
    CWR: 冻结 GNN 共享层，仅重新初始化并微调预测头 (mlp_mean, mlp_var)。
    使用 base 事实回放缓冲区防止知识遗忘。
    参考: Lomonaco & Maltoni, CORe50, CoRL 2017.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model   = model
        self.dataset = dataset
        self.lr      = lr
        self.device  = device
        self.args    = args
        self.steps   = getattr(args, "baseline_steps", 100)

        # 回放缓冲区：从 base_train 采样
        replay_size = getattr(args, "cwr_replay_size", 256)
        labeled_base = _labeled_facts(dataset.base_train)
        self.replay_buffer = random.sample(
            labeled_base, min(replay_size, len(labeled_base))
        ) if labeled_base else []

        # 保存任务结束时的预测头权重快照（"CW" weights）
        self._cw_mean_state = copy.deepcopy(model.mlp_mean.state_dict())
        self._cw_var_state  = copy.deepcopy(model.mlp_var.state_dict())

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        # ── 1. 冻结 GNN 层及 embedding，仅训练预测头 ──
        for name, param in self.model.named_parameters():
            param.requires_grad = ("mlp_mean" in name or "mlp_var" in name)

        # ── 2. 重新初始化预测头 (Re-init) ──
        def _reset_linear(module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        _reset_linear(self.model.mlp_mean)
        _reset_linear(self.model.mlp_var)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)

        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            with torch.no_grad():
                z = self.model(ei, et, ec)

            # 新事实监督
            batch = labeled_new + (random.sample(self.replay_buffer,
                                                  min(len(self.replay_buffer), len(labeled_new)))
                                   if self.replay_buffer else [])
            h_t = torch.tensor([f[0] for f in batch], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in batch], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in batch], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in batch], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)
            loss.backward()
            optimizer.step()

        # ── 3. 合并 CW 权重：当前头权重和 base 快照的加权平均 ──
        alpha = getattr(self.args, "cwr_alpha", 0.5)
        with torch.no_grad():
            for name, param in self.model.mlp_mean.named_parameters():
                cw_val = self._cw_mean_state[name].to(self.device)
                param.copy_(alpha * param + (1 - alpha) * cw_val)
            for name, param in self.model.mlp_var.named_parameters():
                cw_val = self._cw_var_state[name].to(self.device)
                param.copy_(alpha * param + (1 - alpha) * cw_val)

        # 更新快照
        self._cw_mean_state = copy.deepcopy(self.model.mlp_mean.state_dict())
        self._cw_var_state  = copy.deepcopy(self.model.mlp_var.state_dict())

        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PNN (Progressive Neural Networks)
# ─────────────────────────────────────────────────────────────────────────────

class _LateralAdapter(nn.Module):
    """将冻结列的特征映射到当前列的加性修正量。"""

    def __init__(self, in_dim, adapter_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PNNUpdater:
    """
    PNN: 渐进式神经网络。
    每次增量阶段添加横向适配器；冻结旧参数，仅训练新的横向连接。
    参考: Rusu et al., Progressive Neural Networks, 2016.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model        = model
        self.dataset      = dataset
        self.lr           = lr
        self.device       = device
        self.args         = args
        self.steps        = getattr(args, "baseline_steps", 100)
        self.adapter_dim  = getattr(args, "pnn_adapter_dim", 32)
        self.emb_dim      = model.emb_dim
        self.adapters     = nn.ModuleList()  # 横向适配器列表（累积）

        # 保存第一列（base 列）的参数快照
        self._frozen_params = [
            p.detach().clone() for p in model.parameters()
        ]

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        # ── 1. 冻结所有旧参数 ──
        for param in self.model.parameters():
            param.requires_grad = False

        # ── 2. 为当前增量阶段添加一组横向适配器 ──
        # 输入维度为 3 * emb_dim（与 predict() 的拼接一致）
        lateral_in = 3 * self.emb_dim
        new_adapter = _LateralAdapter(
            lateral_in, self.adapter_dim, lateral_in
        ).to(self.device)
        self.adapters.append(new_adapter)

        optimizer = torch.optim.Adam(new_adapter.parameters(), lr=self.lr)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            for param in self.model.parameters():
                param.requires_grad = True
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)

        self.model.eval()
        for _ in range(self.steps):
            optimizer.zero_grad()

            with torch.no_grad():
                z = self.model(ei, et, ec)

            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            r_feat = self.model.relation_emb(r_t)
            h_r_t = torch.cat([z[h_t], r_feat, z[t_t]], dim=-1)

            # 横向修正：累加所有适配器的输出
            correction = sum(adapter(h_r_t.detach()) for adapter in self.adapters)
            h_r_t_adapted = h_r_t.detach() + correction

            # Apply adapted triple representation directly through prediction heads
            mu       = self.model.mlp_mean(h_r_t_adapted).squeeze(-1)
            sigma_sq = self.model.mlp_var(h_r_t_adapted).squeeze(-1)

            loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)
            loss.backward()
            optimizer.step()

        # 解冻模型参数
        for param in self.model.parameters():
            param.requires_grad = True

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SI (Synaptic Intelligence)
# ─────────────────────────────────────────────────────────────────────────────

class SIUpdater:
    """
    SI: 通过路径积分（梯度 × 参数变化量）跟踪参数重要性，
    微调时对重要参数施加二次正则化惩罚。
    参考: Zenke et al., Continual Learning Through Synaptic Intelligence, ICML 2017.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model   = model
        self.dataset = dataset
        self.lr      = lr
        self.device  = device
        self.args    = args
        self.steps   = getattr(args, "baseline_steps", 100)
        self.c       = getattr(args, "si_c", 0.1)
        self.epsilon = getattr(args, "si_epsilon", 0.1)

        # θ* ：上一个任务结束时的参数快照
        self._theta_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        # W：路径积分累积量
        self._W = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad
        }
        # Ω：归一化后的参数重要性
        self._omega = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad
        }
        # 上一步参数值（用于计算 Δθ）
        self._prev_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def _si_penalty(self):
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._omega:
                penalty = penalty + (self._omega[n] * (p - self._theta_star[n]) ** 2).sum()
        return self.c * penalty

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            z = self.model(ei, et, ec)

            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t) + self._si_penalty()
            loss.backward()

            # 累积路径积分 W_k += -g_k * Δθ_k
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in self._W:
                        delta = p.detach() - self._prev_params[n]
                        self._W[n] -= p.grad.detach() * delta
                        self._prev_params[n] = p.detach().clone()

            optimizer.step()

        # ── 更新 Ω（归一化）以及 θ* ──
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self._W:
                    denom = (p.detach() - self._theta_star[n]) ** 2 + self.epsilon
                    self._omega[n] = (self._W[n] / denom).clamp(min=0.0)
                    self._W[n].zero_()
                    self._theta_star[n] = p.detach().clone()
                    self._prev_params[n] = p.detach().clone()

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EWC (Elastic Weight Consolidation)
# ─────────────────────────────────────────────────────────────────────────────

class EWCUpdater:
    """
    EWC: 用 Fisher 信息矩阵（对角近似）约束参数在旧任务上的漂移。
    参考: Kirkpatrick et al., Overcoming Catastrophic Forgetting, PNAS 2017.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model          = model
        self.dataset        = dataset
        self.lr             = lr
        self.device         = device
        self.args           = args
        self.steps          = getattr(args, "baseline_steps", 100)
        self.ewc_lambda     = getattr(args, "ewc_lambda", 5000.0)
        self.fisher_samples = getattr(args, "ewc_fisher_samples", 1024)

        # θ*：base 训练结束时的参数快照
        self._theta_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

        # 计算 Fisher 信息矩阵（对角近似）
        self._fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        labeled_base = _labeled_facts(self.dataset.base_train)
        if not labeled_base:
            return fisher

        sample_facts = random.sample(
            labeled_base, min(self.fisher_samples, len(labeled_base))
        )

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)

        self.model.eval()
        for f in sample_facts:
            self.model.zero_grad()
            with torch.enable_grad():
                z = self.model(ei, et, ec)
                h_t = torch.tensor([f[0]], dtype=torch.long, device=self.device)
                r_t = torch.tensor([f[1]], dtype=torch.long, device=self.device)
                t_t = torch.tensor([f[2]], dtype=torch.long, device=self.device)
                c_t = torch.tensor([f[3]], dtype=torch.float, device=self.device)

                mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
                loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)
                loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

        n_samples = len(sample_facts)
        for n in fisher:
            fisher[n] /= n_samples

        return fisher

    def _ewc_penalty(self):
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                penalty = penalty + (
                    self._fisher[n] * (p - self._theta_star[n]) ** 2
                ).sum()
        return (self.ewc_lambda / 2.0) * penalty

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            z = self.model(ei, et, ec)

            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t) + self._ewc_penalty()
            loss.backward()
            optimizer.step()

        # 更新 Fisher 和 θ*
        self._theta_star = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._fisher = self._compute_fisher()

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 5. GEM (Gradient Episodic Memory)
# ─────────────────────────────────────────────────────────────────────────────

def _project_gradient_gem(grad, ref_grads, margin=0.5):
    """
    GEM 梯度投影：将 grad 投影到不增加任意记忆样本损失的半空间。
    使用 QP 求解（quadprog），若不满足约束则修改梯度。
    """
    if len(ref_grads) == 0:
        return grad

    # G: (n_constraints, n_params)
    G = torch.stack(ref_grads).cpu().double().numpy()  # (M, P)
    t = grad.cpu().double().numpy()                    # (P,)

    # 检查是否满足 G @ g >= -margin（即点积 >= -margin）
    dots = G @ t
    if np.all(dots >= -margin):
        return grad

    # 构造 QP：min 0.5 * ||v - t||^2  s.t.  G @ v >= -margin
    P = G @ G.T                                     # (M, M)
    P = (P + P.T) / 2 + 1e-6 * np.eye(len(P))
    q = G @ t                                       # (M,)
    h_vec = np.full(len(P), margin, dtype=np.float64)
    ones  = np.eye(len(P))

    try:
        v = quadprog.solve_qp(P, q, ones.T, h_vec)[0]  # (M,)
        v_proj = t + G.T @ (v - q)
    except Exception:
        v_proj = t

    return torch.from_numpy(v_proj).float().to(grad.device)


class GEMUpdater:
    """
    GEM: 梯度情景记忆，保持记忆缓冲区中 base 事实的损失不增加。
    参考: Lopez-Paz & Ranzato, GEM, NeurIPS 2017.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model   = model
        self.dataset = dataset
        self.lr      = lr
        self.device  = device
        self.args    = args
        self.steps   = getattr(args, "baseline_steps", 100)
        self.margin  = getattr(args, "gem_margin", 0.5)

        memory_size = getattr(args, "gem_memory_size", 256)
        labeled_base = _labeled_facts(dataset.base_train)
        self.memory = random.sample(
            labeled_base, min(memory_size, len(labeled_base))
        ) if labeled_base else []

    def _get_memory_grads(self, z, ei, et, ec):
        """对记忆样本分别计算梯度，返回扁平化梯度列表。"""
        ref_grads = []
        if not self.memory:
            return ref_grads

        # 每次用全部记忆样本批量计算一个梯度向量
        self.model.zero_grad()
        z_mem = self.model(ei, et, ec)
        h_m = torch.tensor([f[0] for f in self.memory], dtype=torch.long, device=self.device)
        r_m = torch.tensor([f[1] for f in self.memory], dtype=torch.long, device=self.device)
        t_m = torch.tensor([f[2] for f in self.memory], dtype=torch.long, device=self.device)
        c_m = torch.tensor([f[3] for f in self.memory], dtype=torch.float, device=self.device)

        mu_m, sig_m = self.model.predict(z_mem[h_m], r_m, z_mem[t_m])
        loss_m = self.model.heteroscedastic_loss(mu_m, sig_m, c_m)
        loss_m.backward()

        g = torch.cat([
            p.grad.detach().view(-1)
            for p in self.model.parameters()
            if p.grad is not None
        ])
        ref_grads.append(g)
        return ref_grads

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            z = self.model(ei, et, ec)

            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)
            loss.backward()

            # 获取新事实梯度
            new_grad = torch.cat([
                p.grad.detach().view(-1)
                for p in self.model.parameters()
                if p.grad is not None
            ])

            # 计算记忆梯度并投影
            self.model.zero_grad()
            ref_grads = self._get_memory_grads(z.detach(), ei, et, ec)
            projected = _project_gradient_gem(new_grad, ref_grads, self.margin)

            # 将投影梯度写回参数
            self.model.zero_grad()
            offset = 0
            for p in self.model.parameters():
                numel = p.numel()
                p.grad = projected[offset: offset + numel].view_as(p).clone()
                offset += numel

            optimizer.step()

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMR (Embedding Memory Replay)
# ─────────────────────────────────────────────────────────────────────────────

class EMRUpdater:
    """
    EMR: 嵌入记忆回放 + 嵌入对齐损失，防止实体嵌入发生漂移。
    参考: Wang et al., Sentence Embedding Alignment for Lifelong Relation Extraction, NAACL 2019.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model       = model
        self.dataset     = dataset
        self.lr          = lr
        self.device      = device
        self.args        = args
        self.steps       = getattr(args, "baseline_steps", 100)
        self.align_coeff = getattr(args, "emr_align_coeff", 0.1)

        memory_size = getattr(args, "emr_memory_size", 256)
        labeled_base = _labeled_facts(dataset.base_train)
        self.memory_facts = random.sample(
            labeled_base, min(memory_size, len(labeled_base))
        ) if labeled_base else []

        # 存储 base 阶段实体嵌入锚点（克隆固定）
        with torch.no_grad():
            self._anchor_emb = model.entity_emb.weight.detach().clone().to(device)

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            z = self.model(ei, et, ec)

            # ── 新事实监督损失 ──
            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss_new = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)

            # ── 记忆回放损失 ──
            loss_replay = torch.tensor(0.0, device=self.device)
            if self.memory_facts:
                h_m = torch.tensor([f[0] for f in self.memory_facts], dtype=torch.long, device=self.device)
                r_m = torch.tensor([f[1] for f in self.memory_facts], dtype=torch.long, device=self.device)
                t_m = torch.tensor([f[2] for f in self.memory_facts], dtype=torch.long, device=self.device)
                c_m = torch.tensor([f[3] for f in self.memory_facts], dtype=torch.float, device=self.device)

                mu_m, sig_m = self.model.predict(z[h_m], r_m, z[t_m])
                loss_replay = self.model.heteroscedastic_loss(mu_m, sig_m, c_m)

            # ── 嵌入对齐损失（防止 base 实体嵌入漂移）──
            base_ent_count = min(self.dataset.base_num_ent, self.model.entity_emb.weight.shape[0])
            current_emb = self.model.entity_emb.weight[:base_ent_count]
            anchor_emb  = self._anchor_emb[:base_ent_count]
            loss_align  = F.mse_loss(current_emb, anchor_emb)

            loss = loss_new + loss_replay + self.align_coeff * loss_align
            loss.backward()
            optimizer.step()

        # 更新嵌入锚点（包含新实体）
        with torch.no_grad():
            self._anchor_emb = self.model.entity_emb.weight.detach().clone()

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DiCGRL (Disentangle-based Continual Graph Representation Learning)
# ─────────────────────────────────────────────────────────────────────────────

class DiCGRLUpdater:
    """
    DiCGRL: 将实体嵌入解耦为 K 个独立子空间；
    增量时通过注意力路由识别活跃子空间，仅更新活跃子空间；
    对非活跃子空间施加知识巩固正则化。
    参考: Liu et al., DiCGRL, IJCAI 2021.
    """

    def __init__(self, model, dataset, lr, gamma, device, args):
        self.model       = model
        self.dataset     = dataset
        self.lr          = lr
        self.device      = device
        self.args        = args
        self.steps       = getattr(args, "baseline_steps", 100)
        self.K           = getattr(args, "dicgrl_num_subspaces", 4)
        self.consol_coef = getattr(args, "dicgrl_consolidation_coeff", 0.1)

        emb_dim = model.emb_dim
        assert emb_dim % self.K == 0, (
            f"emb_dim ({emb_dim}) must be divisible by dicgrl_num_subspaces ({self.K})"
        )
        self.sub_dim = emb_dim // self.K

        # 路由注意力网络：输入 triple 特征，输出每个子空间的软路由分数
        self.router = nn.Linear(3 * emb_dim, self.K).to(device)

        # 保存旧参数快照（用于巩固损失）
        self._old_entity_emb = model.entity_emb.weight.detach().clone()

    def _subspace_mask(self, routing_scores):
        """
        根据路由分数确定活跃子空间掩码（topK/2 个子空间视为活跃）。
        routing_scores: (N, K) → 返回 (K,) bool 掩码
        """
        avg_scores = routing_scores.mean(0)          # (K,)
        topk = max(1, self.K // 2)
        _, active_idx = avg_scores.topk(topk)
        mask = torch.zeros(self.K, dtype=torch.bool, device=self.device)
        mask[active_idx] = True
        return mask

    def step(self, new_facts_batch):
        _init_new_entities(self.model, self.dataset, self.device)

        labeled_new = _labeled_facts(new_facts_batch)
        if not labeled_new:
            _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
            return _step_summary(self.model, self.dataset, new_facts_batch, self.device)

        ei, et, ec = _base_graph_tensors(self.dataset, self.device)

        # 计算路由分数，确定活跃子空间
        self.model.eval()
        with torch.no_grad():
            z = self.model(ei, et, ec)
            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            r_feat = self.model.relation_emb(r_t)
            hrt = torch.cat([z[h_t], r_feat, z[t_t]], dim=-1)
            routing_scores = torch.softmax(self.router(hrt), dim=-1)  # (N, K)

        active_mask = self._subspace_mask(routing_scores)  # (K,) bool

        # 构造 embedding 梯度掩码：仅允许活跃子空间对应维度的梯度流通
        # 子空间 k 对应维度 [k*sub_dim, (k+1)*sub_dim)
        grad_mask = torch.zeros(self.model.entity_emb.embedding_dim, device=self.device)
        for k in range(self.K):
            if active_mask[k]:
                grad_mask[k * self.sub_dim: (k + 1) * self.sub_dim] = 1.0

        params = list(self.model.parameters()) + list(self.router.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        self.model.train()
        self.router.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            z = self.model(ei, et, ec)

            h_t = torch.tensor([f[0] for f in labeled_new], dtype=torch.long, device=self.device)
            r_t = torch.tensor([f[1] for f in labeled_new], dtype=torch.long, device=self.device)
            t_t = torch.tensor([f[2] for f in labeled_new], dtype=torch.long, device=self.device)
            c_t = torch.tensor([f[3] for f in labeled_new], dtype=torch.float, device=self.device)

            mu, sigma_sq = self.model.predict(z[h_t], r_t, z[t_t])
            loss_pred = self.model.heteroscedastic_loss(mu, sigma_sq, c_t)

            # ── 知识巩固损失（非活跃子空间保持不变）──
            loss_consol = torch.tensor(0.0, device=self.device)
            base_ent_count = min(self.dataset.base_num_ent, self.model.entity_emb.weight.shape[0])
            if base_ent_count > 0:
                old_emb = self._old_entity_emb[:base_ent_count].to(self.device)
                cur_emb = self.model.entity_emb.weight[:base_ent_count]

                for k in range(self.K):
                    if not active_mask[k]:
                        s = k * self.sub_dim
                        e = (k + 1) * self.sub_dim
                        loss_consol = loss_consol + F.mse_loss(
                            cur_emb[:, s:e], old_emb[:, s:e]
                        )

            loss = loss_pred + self.consol_coef * loss_consol
            loss.backward()

            # 抑制非活跃子空间的梯度
            if self.model.entity_emb.weight.grad is not None:
                self.model.entity_emb.weight.grad *= grad_mask.unsqueeze(0)

            optimizer.step()

        # 更新旧参数快照
        self._old_entity_emb = self.model.entity_emb.weight.detach().clone()

        _update_belief_state(self.model, self.dataset, new_facts_batch, self.device)
        return _step_summary(self.model, self.dataset, new_facts_batch, self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "cwr":     CWRUpdater,
    "pnn":     PNNUpdater,
    "si":      SIUpdater,
    "ewc":     EWCUpdater,
    "gem":     GEMUpdater,
    "emr":     EMRUpdater,
    "dicgrl":  DiCGRLUpdater,
}


def get_baseline(name, model, dataset, lr, gamma, device, args):
    """根据名称实例化对应的基线更新器。"""
    name = name.lower()
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"未知的基线模型 '{name}'，可选: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name](model, dataset, lr, gamma, device, args)

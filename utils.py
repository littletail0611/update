# 工具类：评估指标(MSE, MAE等)、日志记录

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """计算回归任务的核心评估指标：MSE, MAE, RMSE"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # 如果验证集为空，直接返回 0 防止报错
    if len(y_true) == 0:
        return {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0}
        
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse
    }

def evaluate_model(model, test_data, z, device="cpu"):
    """
    在给定的测试集上评估模型
    加入了参数 z: 必须传入经过 GNN 聚合后的实体特征，而不是原始 Embedding！
    只评估有已知置信度（第4个元素不为 None）的事实。
    """
    if not test_data:
        return calculate_metrics([], [])

    # 过滤掉无标注事实（置信度为 None）
    labeled_data = [fact for fact in test_data if fact[3] is not None]
    if not labeled_data:
        return calculate_metrics([], [])

    model.eval() 
    
    h_idx = torch.tensor([fact[0] for fact in labeled_data], dtype=torch.long).to(device)
    r_idx = torch.tensor([fact[1] for fact in labeled_data], dtype=torch.long).to(device)
    t_idx = torch.tensor([fact[2] for fact in labeled_data], dtype=torch.long).to(device)
    y_true = torch.tensor([fact[3] for fact in labeled_data], dtype=torch.float) 
    
    with torch.no_grad():
        # 【核心修复】：使用传入的 GNN 聚合特征 z，而不是 model.entity_emb
        z_h = z[h_idx]
        z_t = z[t_idx]
        
        mu_pred, _ = model.predict(z_h, r_idx, z_t)
        
    return calculate_metrics(y_true, mu_pred.cpu())

def evaluate_belief_state(dataset, test_data):
    """评估系统当前维护的全局 belief state 与真实标签的差异"""
    if not test_data:
        return calculate_metrics([], [])
        
    y_true = []
    y_pred = []
    
    for h_id, r_id, t_id, c_true in test_data:
        fact_tuple = (h_id, r_id, t_id)
        # 如果系统中存在该事实的 belief，则提取；否则默认给 0.5 作为中立猜测
        c_pred = dataset.belief_state.get(fact_tuple, 0.5)
        
        y_true.append(c_true)
        y_pred.append(c_pred)
        
    return calculate_metrics(y_true, y_pred)

class Logger:
    @staticmethod
    def print_metrics(stage_name, metrics):
        print(f"[{stage_name} Evaluation]")
        print(f"  - MSE:  {metrics['MSE']:.4f}")
        print(f"  - MAE:  {metrics['MAE']:.4f}")
        print(f"  - RMSE: {metrics['RMSE']:.4f}")
        print("-" * 30)
# 第一阶段：在 base 数据上离线训练 Base 模型

import torch
import torch.optim as optim
import os

from dataset import UKGDataset
from model import HeteroscedasticBaseModel
from utils import evaluate_model, Logger
from config import get_args

def train_base(args):
    print(f"=== 第一阶段: 启动 Base 模型离线训练 (设备: {args.device}) ===")
    
    # 动态提取数据集名称，拼接保存路径
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    ckpt_path = f"checkpoints/base_model_{dataset_name}.pth"
    print(f">>> 当前数据集: {dataset_name} | 权重将保存至: {ckpt_path}")
    
    # 1. 加载数据集 (报错就是因为之前不小心把这三行删了！)
    dataset = UKGDataset(data_dir=args.data_dir)
    num_ent = dataset.num_ent
    num_rel = dataset.num_rel
    
    # 2. 初始化 Base 模型并挂载到指定设备
    model = HeteroscedasticBaseModel(
        num_entities=num_ent, 
        num_relations=num_rel, 
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.base_weight_decay)
    
    # 获取 Base 图的结构数据并挂载到设备
    edge_index, edge_type, edge_conf_target = dataset.get_base_graph_data()
    edge_index = edge_index.to(args.device)
    edge_type = edge_type.to(args.device)
    edge_conf_target = edge_conf_target.to(args.device)
    
    num_edges = edge_index.shape[1]
    print(f"开始训练: 实体数={num_ent}, 关系数={num_rel}, 边数={num_edges}")
    
    best_valid_mse = float('inf')
    patience_counter = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    # 3. 训练循环 (Mini-batch + 对比学习)
    for epoch in range(args.base_epochs):
        model.train()
        
        # 随机打乱边的索引
        perm = torch.randperm(num_edges, device=args.device)
        epoch_loss_conf = 0
        epoch_loss_cl = 0
        
        # 遍历各个 Batch
        for i in range(0, num_edges, args.base_batch_size):
            optimizer.zero_grad()
            
            # 获取当前 Batch 在全局边集中的索引
            batch_idx = perm[i:i + args.base_batch_size]
            
            # ==========================================
            # 【新增】：Edge Masking (边掩码) 核心逻辑
            # ==========================================
            # 1. 创建一个全为 True 的布尔掩码
            mask = torch.ones(num_edges, dtype=torch.bool, device=args.device)
            # 2. 将当前 Batch 要预测的边设为 False (屏蔽掉)
            mask[batch_idx] = False
            
            # 3. 构造只包含“上下文边”的残缺图
            masked_edge_index = edge_index[:, mask]
            masked_edge_type = edge_type[mask]
            masked_edge_conf = edge_conf_target[mask]
            
            # 4. GNN 前向传播：在残缺图上进行消息传递获取实体特征 z
            # 此时目标边的置信度被完全隔绝，彻底杜绝标签泄露
            z = model(masked_edge_index, masked_edge_type, masked_edge_conf)
            # ==========================================
            
            # 5. 提取目标边的头尾节点特征，进行预测
            b_h = z[edge_index[0, batch_idx]]
            b_t = z[edge_index[1, batch_idx]]
            b_r = edge_type[batch_idx]
            b_conf = edge_conf_target[batch_idx]
            
            # 1. 异方差预测损失
            mu, sigma_sq = model.predict(b_h, b_r, b_t)
            loss_conf = model.heteroscedastic_loss(mu, sigma_sq, b_conf)
            
            # 2. 连续标签感知的对比学习损失
            loss_cl = model.continuous_contrastive_loss(b_h, b_r, b_t, b_conf)
            
            # 3. 联合优化
            loss = loss_conf + args.alpha_cl * loss_cl
            loss.backward()
            optimizer.step()
            
            epoch_loss_conf += loss_conf.item()
            epoch_loss_cl += loss_cl.item()
    
        # 每 2 个 Epoch 评估一次验证集，执行早停判定
        if (epoch + 1) % 2 == 0:
            num_batches = (num_edges // args.base_batch_size) + 1
            avg_conf_loss = epoch_loss_conf / num_batches
            avg_cl_loss = epoch_loss_cl / num_batches
            
            # 注意：传入评估需要用全局最优的 z，所以在 eval 前重新过一次全图
            model.eval()
            with torch.no_grad():
                current_z = model(edge_index, edge_type, edge_conf_target)
            
            valid_metrics = evaluate_model(model, dataset.base_valid, z=current_z, device=args.device)
            current_mse = valid_metrics['MSE']
            current_mae = valid_metrics['MAE']
            
            print(f"Epoch [{epoch+1}/{args.base_epochs}] | L_conf: {avg_conf_loss:.4f} | L_cl: {avg_cl_loss:.4f} | Valid MSE: {current_mse:.4f} | Valid MAE: {current_mae:.4f}")
            
            # 早停机制
            if current_mse < best_valid_mse:
                best_valid_mse = current_mse
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print("  --> [模型提升] 最佳权重已保存")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n触发早停 (Early Stopping)! 验证集已连续 {args.patience * 2} 轮未提升。")
                    break

    # 4. 最终测试集评估
    print("\n=== Base 模型训练完成，加载最佳权重执行最终评估 ===")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        best_z = model(edge_index, edge_type, edge_conf_target)
    
    test_metrics = evaluate_model(model, dataset.base_test, z=best_z, device=args.device)
    Logger.print_metrics("Base Test Set", test_metrics)
    return test_metrics

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_base(args)
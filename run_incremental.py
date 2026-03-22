# 第二阶段：加载训练好的 Base 模型，在 inc 数据上执行 belief 更新

import torch
import os
import random

from dataset import UKGDataset
from model import HeteroscedasticBaseModel
from updater import UnifiedConfidenceUpdater
from utils import evaluate_model, Logger
from config import get_args

def run_incremental_update(args):
    print(f"=== 第二阶段: 启动流式增量 Belief 更新 (设备: {args.device}) ===")
    
    # 动态提取数据集名称，匹配对应的权重路径
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    ckpt_path = f"checkpoints/base_model_{dataset_name}.pth"
    final_belief_path = f"checkpoints/final_belief_state_{dataset_name}.pt"
    
    # 1. 加载包含 Inc 数据的数据集实例
    dataset = UKGDataset(data_dir=args.data_dir)
    
    # 初始化模型
    model = HeteroscedasticBaseModel(
        num_entities=dataset.num_ent, 
        num_relations=dataset.num_rel, 
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    
    # 2. 从动态路径加载 Base 模型权重
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到预训练模型 {ckpt_path}，请先运行 train_base.py")
        
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device), strict=False)
    model.eval()
    print(f">>> 成功加载 Base 模型权重 ({ckpt_path})。")
    
    # 获取 Base 图的结构数据
    edge_index, edge_type, edge_conf = dataset.get_base_graph_data()
    edge_index = edge_index.to(args.device)
    edge_type = edge_type.to(args.device)
    edge_conf = edge_conf.to(args.device)
    
    # [Baseline] 增量更新前的初始表现评估
    with torch.no_grad():
        init_z = model(edge_index, edge_type, edge_conf)
        
    print("\n[Baseline] 增量更新前的初始表现评估:")
    init_inc_valid = evaluate_model(model, dataset.inc_valid, z=init_z, device=args.device)
    Logger.print_metrics("Pre-update Inc Valid", init_inc_valid)
    
    # 3. 初始化统一置信度更新器
    updater = UnifiedConfidenceUpdater(
        model=model, 
        dataset=dataset, 
        lr=args.inc_lr, 
        gamma=args.gamma,
        device=args.device,
        args=args 
    )
    
    # 4. 模拟流式处理新事实 Batch
    print(f"\n>>> 开始处理 Inc 新事实 (Batch Size: {args.inc_batch_size})...")
    inc_batches = list(dataset.get_incremental_batches(batch_size=args.inc_batch_size))
    
    for batch_idx, batch_facts in enumerate(inc_batches):
        print(f"\n--- 处理 Batch {batch_idx + 1}/{len(inc_batches)} ---")
        
        # 执行核心更新逻辑，返回真实的变动幅度
        # new_mu, actual_change = updater.step(batch_facts)
        new_mu, actual_change_mean, actual_change_max, affected_count = updater.step(batch_facts)
        
        print(f"Batch {batch_idx + 1} 摘要:")
        print(f"  - 推断新事实平均置信度: {new_mu.mean().item():.4f}")
        print(f"  - 全局平均变动幅度: {actual_change_mean:.6f}")
        print(f"  - 局部最大变动幅度: {actual_change_max:.4f}")
        print(f"  - 达到阈值受影响的旧事实数: {affected_count} 条")
        
    print("\n=== 所有增量更新任务执行完毕，执行最终全面评估 ===")
    
    # 5. 评估增量更新后的整体表现
    model.eval()
    with torch.no_grad():
        final_z = model(edge_index, edge_type, edge_conf)
    
    # 评估 1：增量更新后的新知识掌握情况 (模型有没有学到新东西)
    inc_test_metrics = evaluate_model(model, dataset.inc_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update Model on Inc Test", inc_test_metrics)
    
    # 评估 2：【核心修复】用相同的神经网络推断函数，评估旧知识记忆情况 (验证灾难性遗忘)
    base_test_metrics = evaluate_model(model, dataset.base_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update Model on Base Test", base_test_metrics)

    # ==========================================
    # 评估 3：在 Base 和 Inc 的测试集并集上进行综合评估
    print("\n[Joint Evaluation] 进行 Base 和 Inc 联合测试集评估...")
    
    # 根据数据类型自动选择拼接方式
    if isinstance(dataset.base_test, torch.Tensor):
        combined_test = torch.cat([dataset.base_test, dataset.inc_test], dim=0)
    elif isinstance(dataset.base_test, list):
        combined_test = dataset.base_test + dataset.inc_test
    else:
        # 假设是 Numpy Array 或是其他支持 numpy 接口的数据格式
        import numpy as np
        combined_test = np.concatenate([dataset.base_test, dataset.inc_test], axis=0)

    # 在并集上执行评估
    combined_test_metrics = evaluate_model(model, combined_test, z=final_z, device=args.device)
    Logger.print_metrics("Post-update Model on Combined Test (Base + Inc)", combined_test_metrics)
    # ==========================================
    
    # 6. 保存最终的全局 Belief 状态 (带数据集后缀)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(dataset.belief_state, final_belief_path)
    print(f">>> 全局 Belief 状态已保存至 {final_belief_path}")

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_incremental_update(args)
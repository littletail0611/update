import os
import time
import torch
import argparse
import pandas as pd

# 导入您自己编写的模块
from dataset import UKGDataset
from model import HeteroscedasticBaseModel
from utils import evaluate_model, Logger
from updater import UnifiedConfidenceUpdater

def run_ablation_experiments(args):
    """
    统筹运行所有的消融实验变体，并生成对比表格
    """
    ablation_modes = [
        "full",          # 完整模型 (Ours)
        "wo_causal",     # 变体 1: 移除受影响子图判定
        "wo_bayes",      # 变体 2: 移除贝叶斯信念融合
        "wo_kd",         # 变体 3: 移除未受影响区域全局 KD 保护
        "wo_em_freeze",  # 变体 4: 移除 EM 阶段基座物理冻结
        "wo_link_pred",  # 变体 5: 移除链接预测自监督损失
    ]
    
    results = []
    
    # 根据数据集名称动态决定要加载的基础权重路径
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    base_weight_path = f"checkpoints/base_model_{dataset_name}.pth"

    print("=" * 80)
    print(f"🚀 启动自动化消融实验 | 数据集: {dataset_name}")
    print("=" * 80)

    for mode in ablation_modes:
        print(f"\n>>> [当前进行] 变体: {mode.upper()} ...")
        start_time = time.time()
        
        # 1. 【核心防御】每次实验必须重新加载“干净且初始”的数据集和模型
        # 防止上一个消融变体污染了 belief_state 或权重
        dataset = UKGDataset(data_dir=args.data_dir)
        
        model = HeteroscedasticBaseModel(
            num_entities=dataset.num_ent, 
            num_relations=dataset.num_rel, 
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
        
        # 尝试加载 Base 阶段训练好的干净权重
        try:
            model.load_state_dict(torch.load(base_weight_path, map_location=args.device))
        except FileNotFoundError:
            print(f"❌ 找不到 Base 模型权重: {base_weight_path}，请先运行 train_base.py！")
            return
            
        model.to(args.device)
        
        # 2. 初始化带特定消融模式的 Updater
        updater = UnifiedConfidenceUpdater(
            model=model, 
            dataset=dataset, 
            lr=args.lr, 
            gamma=args.gamma,
            device=args.device, 
            args=args,
            ablation_mode=mode  # <--- 将消融模式注入到更新器
        )
        
        # 3. 模拟增量更新数据流 (遍历所有 Batch)
        inc_batches = dataset.get_incremental_batches(batch_size=args.batch_size)
        total_affected = 0
        total_batches = len(dataset.inc_train) // args.batch_size + (1 if len(dataset.inc_train) % args.batch_size != 0 else 0)
        
        for i, batch in enumerate(inc_batches):
            new_mu, mean_chg, max_chg, affected = updater.step(batch)
            total_affected += affected
            
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                print(f"    - 处理 Batch {i+1}/{total_batches} | 本批次跨越阈值受影响边数: {affected}")
                
        # 4. 执行最终模型评测 (评估新事实推理和防遗忘能力)
        print("    --> 更新完成，正在执行全量评估...")
        
        # 获取最新的 Base 图谱数据并跑一次 GNN 聚合得到 z
        # 注意：评测函数需要 z (聚合后的特征)，不能直接传原始 embedding
        model.eval()
        with torch.no_grad():
            edge_idx, edge_type, edge_conf = dataset.get_base_graph_data()
            edge_idx = edge_idx.to(args.device)
            edge_type = edge_type.to(args.device)
            edge_conf = edge_conf.to(args.device)
            z_final = model(edge_idx, edge_type, edge_conf)
        
        # 分别在 Inc Test 和 Base Test 上评测
        inc_metrics = evaluate_model(model, dataset.inc_test, z_final, args.device)
        base_metrics = evaluate_model(model, dataset.base_test, z_final, args.device)
        
        inc_mse, inc_mae = inc_metrics['MSE'], inc_metrics['MAE']
        base_mse, base_mae = base_metrics['MSE'], base_metrics['MAE']
        
        time_cost = time.time() - start_time
        print(f"  ✅ {mode.upper()} 完成 | 耗时: {time_cost:.1f}s | Inc MSE: {inc_mse:.4f} | Base MSE: {base_mse:.4f}")
        
        # 5. 保存结果
        results.append({
            "Model Variant": mode,
            "Inc Test MSE (↓)": round(inc_mse, 4),
            "Inc Test MAE (↓)": round(inc_mae, 4),
            "Base Test MSE (↓)": round(base_mse, 4),
            "Total Affected Edges": total_affected,
            "Time Cost (s)": round(time_cost, 1)
        })

    # ==========================================
    # 打印最终对比表格 (直接复制进论文)
    # ==========================================
    df = pd.DataFrame(results)
    
    # 映射为正式的学术论文术语
    df["Model Variant"] = df["Model Variant"].map({
        "full": "Ours (Full Model)",
        "wo_causal": "w/o Causal Filter",
        "wo_bayes": "w/o Bayesian Filter",
        "wo_kd": "w/o Global KD Protect",
        "wo_em_freeze": "w/o EM Base Freeze",
        "wo_link_pred": "w/o Link Pred Self-Sup",
    })
    
    print("\n" + "=" * 80)
    print("📊 Ablation Study Results (消融实验结果汇总)")
    print("=" * 80)
    # 打印 Markdown 格式的表格
    print(df.to_string(index=False))
    print("=" * 80)
    
    # 输出 CSV 以便后续做条形图或折线图
    csv_name = f"ablation_results_{dataset_name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n✨ 详细结果已保存至: {csv_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Studies for Unified Confidence Updater")
    
    # 数据与环境参数
    parser.add_argument("--data_dir", type=str, default="./datasets/nl27k_ind/", help="Dataset directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # 模型架构参数
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate in MLPs")
    
    # 更新策略与超参数
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for incremental updates")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for refinement")
    parser.add_argument("--gamma", type=float, default=0.8, help="Causal influence decay factor")
    parser.add_argument("--influence_threshold", type=float, default=0.03, help="Threshold for causality filter")
    parser.add_argument("--em_steps", type=int, default=10, help="Steps for EM inference")
    parser.add_argument("--refine_steps", type=int, default=3, help="Steps for local refinement")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="L2 regularization weight")
    parser.add_argument("--alpha_cl", type=float, default=0.5, help="Contrastive loss weight")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="Small constant for numerical stability")
    
    args = parser.parse_args()
    
    # 为了保证公平客观的对比，消融实验请确保使用固定的随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    run_ablation_experiments(args)
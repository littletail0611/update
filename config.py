import argparse
import torch

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="UKG-Updater: Dynamic Confidence Updating")

    # ================= 1. 数据与环境配置 =================
    parser.add_argument("--data_dir", type=str, default="datasets/nl27k_ind", help="数据集根目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证实验可复现")

    # ================= 2. 模型结构参数 =================
    parser.add_argument("--emb_dim", type=int, default=128, help="实体和关系的嵌入维度 (d)")
    parser.add_argument("--num_layers", type=int, default=2, help="Confidence-Aware GNN 的层数 (L)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="防过拟合的 Dropout 比例")

    # ================= 3. Base 模型离线训练参数 =================
    parser.add_argument("--base_epochs", type=int, default=200, help="Base 模型最大训练轮数")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base 模型学习率")
    parser.add_argument("--base_weight_decay", type=float, default=1e-4, help="L2 正则化权重 (防过拟合)")
    parser.add_argument("--base_batch_size", type=int, default=2048, help="Base 模型 Mini-batch 边采样大小")
    parser.add_argument("--alpha_cl", type=float, default=0.5, help="连续对比学习损失的权重")
    parser.add_argument("--patience", type=int, default=10, help="早停机制：容忍多少次验证集评估不下降")
    
    # ================= 4. 增量更新 (Online Updating) 核心数学参数 =================
    parser.add_argument("--inc_batch_size", type=int, default=1024, help="流式处理新事实的 Batch 大小")
    parser.add_argument("--inc_lr", type=float, default=0.005, help="局部 EM 和微调时的学习率")
    parser.add_argument("--em_steps", type=int, default=10, help="局部 EM 推断的交替迭代次数")
    parser.add_argument("--gamma", type=float, default=0.8, help="因果影响多跳传播的衰减因子")
    parser.add_argument("--num_hops", type=int, default=2, help="因果影响传播的最大跳数")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="贝叶斯滤波中防止除零的平滑项")
    parser.add_argument("--influence_threshold", type=float, default=0.03, help="判定旧事实受影响的阈值")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="防止灾难性遗忘的锚点正则化权重")
    parser.add_argument("--refine_steps", type=int, default=3, help="受影响局部子图的微调步数")

    # ================= 5. 传播-微调超参数 =================
    parser.add_argument("--edge_drop_rate", type=float, default=0.2, help="Edge dropout ratio (deprecated: kept for backward compatibility, no longer used)")
    parser.add_argument("--alpha_base", type=float, default=1.0, help="Weight for base outer-constraint loss")
    parser.add_argument("--mlp_anchor_coeff", type=float, default=0.01, help="MLP anchor regularization coefficient")
    parser.add_argument("--finetune_steps", type=int, default=5, help="Fine-tuning steps in propagate-then-finetune stage")
    parser.add_argument("--propagation_hops", type=int, default=2, help="Number of hops for label propagation")
    parser.add_argument("--soft_label_weight", type=float, default=0.5, help="Weight for sigma-weighted soft supervision loss on new facts")
    parser.add_argument("--dynamic_update_interval", type=int, default=2, help="Update pseudo-label every N finetune steps (0 to disable)")
    parser.add_argument("--func_anchor_ratio", type=float, default=0.9, help="Blend ratio for functional anchoring: func_anchor_ratio * functional_loss + (1 - func_anchor_ratio) * weak_absolute_L2")

    # ================= 6. 调参模式 =================
    parser.add_argument("--tuning_mode", action="store_true", default=False, help="Enable tuning mode: suppresses verbose console output during hyperparameter search")

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = get_args()
    print("当前配置参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
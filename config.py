import argparse
import torch

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="UKG-Updater: Dynamic Confidence Updating")

    # ================= 1. 数据与环境配置 =================
    parser.add_argument("--data_dir", type=str, default="./datasets/nl27k_ind", help="数据集根目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证实验可复现")

    # ================= 2. 模型结构参数 =================
    parser.add_argument("--emb_dim", type=int, default=128, help="实体和关系的嵌入维度 (d)")
    parser.add_argument("--num_layers", type=int, default=2, help="Confidence-Aware GNN 的层数 (L)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="防过拟合的 Dropout 比例")

    # ================= 3. Base 模型离线训练参数 =================
    parser.add_argument("--base_epochs", type=int, default=200, help="Base 模型最大训练轮数")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base 模型学习率")
    parser.add_argument("--base_weight_decay", type=float, default=1e-4, help="L2 正则化权重")
    parser.add_argument("--base_batch_size", type=int, default=2048, help="Base 模型 Batch 大小")
    parser.add_argument("--alpha_cl", type=float, default=0.5, help="对比学习损失的权重")
    parser.add_argument("--patience", type=int, default=10, help="早停机制容忍次数")
    
    # ================= 4. 增量更新 (Online Updating) 参数 =================
    # [通用控制]
    parser.add_argument("--inc_batch_size", type=int, default=1024, help="处理新事实的 Batch 大小")
    parser.add_argument("--single_batch", action="store_true", default=True, help="是否将所有增量事实作为一个 Batch 一次性更新")
    parser.add_argument("--no-single_batch", dest="single_batch", action="store_false")
    parser.add_argument("--inc_lr", type=float, default=0.005, help="增量更新器 (Updater) 的学习率")
    parser.add_argument("--mlp_anchor_coeff", type=float, default=0.01, help="全局预测头(MLP)的正则化系数，防止全局预测能力崩溃")

    # [Stage 1: 先立锚再扩散 (Propagate & Finetune)]
    parser.add_argument("--anchor_steps", type=int, default=100, help="Sub-stage 1.1: 纯有标签数据的立锚微调步数")
    parser.add_argument("--finetune_steps", type=int, default=50, help="Sub-stage 1.2: 引入无标签数据的扩散微调步数")
    parser.add_argument("--lambda_ent_reg", type=float, default=1.0, help="立锚期防止新实体表征坍塌的空间正则化权重")
    parser.add_argument("--alpha_labeled_supervision", type=float, default=1.0, help="有标签数据的监督权重")
    parser.add_argument("--alpha_new_supervision", type=float, default=3.0, help="无标签伪标签数据的监督权重")
    parser.add_argument("--dynamic_update_interval", type=int, default=5, help="扩散期更新伪标签的频率(步数)")

    # [Stage 2: 局部因果推断与贝叶斯精炼 (Causal & Refinement)]
    parser.add_argument("--causal_num_hops", type=int, default=2, help="因果影响评估截取的局部子图跳数 (K-hop)")
    parser.add_argument("--gamma", type=float, default=0.8, help="因果影响计算中的全局衰减因子")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="贝叶斯滤波防除零的平滑项")
    parser.add_argument("--influence_threshold", type=float, default=0.005, help="判定旧事实受因果影响的阈值")
    parser.add_argument("--refine_steps", type=int, default=3, help="受影响局部子图的联合微调步数")
    parser.add_argument("--lambda_reg", type=float, default=0.001, help="受影响旧实体的 L2 正则化权重")
    parser.add_argument("--func_anchor_ratio", type=float, default=0.9, help="功能性锚定损失的混合比例")

    # ================= 5. 基线模型超参数 =================
    # [EWC]
    parser.add_argument("--ewc_lambda", type=float, default=5000.0, help="EWC 正则化强度 λ")
    parser.add_argument("--ewc_fisher_samples", type=int, default=1024, help="计算 Fisher 信息矩阵时采样的 Base 事实数量")

    # [SI]
    parser.add_argument("--si_c", type=float, default=0.1, help="SI 替代损失的正则化强度 c")
    parser.add_argument("--si_epsilon", type=float, default=0.1, help="SI 重要性权重归一化的平滑项 ε")

    # [GEM]
    parser.add_argument("--gem_memory_size", type=int, default=256, help="GEM 情景记忆缓冲区大小 (base 事实数)")
    parser.add_argument("--gem_margin", type=float, default=0.5, help="GEM 梯度投影的松弛边距 γ")

    # [EMR]
    parser.add_argument("--emr_memory_size", type=int, default=256, help="EMR 嵌入记忆回放的缓冲区大小")
    parser.add_argument("--emr_align_coeff", type=float, default=0.1, help="EMR 嵌入对齐损失的权重系数")

    # [CWR]
    parser.add_argument("--cwr_replay_size", type=int, default=256, help="CWR 回放缓冲区大小 (base 事实数)")
    parser.add_argument("--cwr_alpha", type=float, default=0.5, help="CWR 当前任务权重与 base 快照的混合比例")

    # [PNN]
    parser.add_argument("--pnn_adapter_dim", type=int, default=32, help="PNN 横向适配器的隐层维度")

    # [DiCGRL]
    parser.add_argument("--dicgrl_num_subspaces", type=int, default=4, help="DiCGRL 解耦子空间数量 K")
    parser.add_argument("--dicgrl_consolidation_coeff", type=float, default=0.1, help="DiCGRL 非活跃子空间的知识巩固损失权重")

    # [通用基线微调步数]
    parser.add_argument("--baseline_steps", type=int, default=100, help="基线模型增量微调步数")

    # ================= 7. 调参模式 =================
    parser.add_argument("--tuning_mode", action="store_true", default=False, help="启用调参模式 (抑制控制台冗余输出)")

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = get_args()
    print("当前配置参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

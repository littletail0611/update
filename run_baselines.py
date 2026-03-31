# run_baselines.py
# 基线模型统一运行脚本：与 run_incremental.py 流程相同，
# 通过 --baseline 参数选择要运行的持续学习基线。

import argparse
import os
import random
import torch

from dataset import UKGDataset
from model import HeteroscedasticBaseModel
from baselines import get_baseline, BASELINE_REGISTRY
from utils import evaluate_model, Logger
from config import get_args


def run_baseline(args):
    baseline_name = args.baseline
    mode_str = "Single-batch (one-shot)" if args.single_batch else "Streaming (mini-batch)"
    print(
        f"=== 基线实验: [{baseline_name.upper()}] | "
        f"模式: {mode_str} | 设备: {args.device} ==="
    )

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    ckpt_path = f"checkpoints/base_model_{dataset_name}.pth"

    # 1. 加载数据集
    dataset = UKGDataset(data_dir=args.data_dir)

    # 2. 初始化模型
    model = HeteroscedasticBaseModel(
        num_entities=dataset.num_ent,
        num_relations=dataset.num_rel,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
    ).to(args.device)

    # 3. 加载预训练 Base 模型权重
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到预训练模型 {ckpt_path}，请先运行 train_base.py"
        )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=args.device), strict=False
    )
    model.eval()
    print(f">>> 成功加载 Base 模型权重 ({ckpt_path})。")

    # Base 图张量（用于增量前的初始评估）
    edge_index, edge_type, edge_conf = dataset.get_base_graph_data()
    edge_index = edge_index.to(args.device)
    edge_type  = edge_type.to(args.device)
    edge_conf  = edge_conf.to(args.device)

    # [Baseline] 增量更新前的初始表现评估
    with torch.no_grad():
        init_z = model(edge_index, edge_type, edge_conf)
    print("\n[Baseline] 增量更新前的初始表现评估:")
    init_inc_valid = evaluate_model(model, dataset.inc_valid, z=init_z, device=args.device)
    Logger.print_metrics("Pre-update Inc Valid", init_inc_valid)

    # 4. 初始化基线更新器
    updater = get_baseline(
        name=baseline_name,
        model=model,
        dataset=dataset,
        lr=args.inc_lr,
        gamma=args.gamma,
        device=args.device,
        args=args,
    )

    # 5. 处理增量事实
    if args.single_batch:
        all_facts = dataset.inc_train
        print(
            f"\n>>> [Single-batch mode] 处理全部 {len(all_facts)} 条增量事实..."
        )
        new_mu, change_mean, change_max, affected_count = updater.step(all_facts)
        print("Single-batch update 摘要:")
        print(f"  - 预测新事实平均置信度: {new_mu.mean().item():.4f}")
        print(f"  - 全局平均变动幅度: {change_mean:.6f}")
        print(f"  - 局部最大变动幅度: {change_max:.4f}")
        print(f"  - 达到阈值受影响的旧事实数: {affected_count}")
    else:
        print(f"\n>>> 开始处理增量事实 (Batch Size: {args.inc_batch_size})...")
        inc_batches = list(dataset.get_incremental_batches(batch_size=args.inc_batch_size))
        for batch_idx, batch_facts in enumerate(inc_batches):
            print(f"\n--- 处理 Batch {batch_idx + 1}/{len(inc_batches)} ---")
            new_mu, change_mean, change_max, affected_count = updater.step(batch_facts)
            print(f"Batch {batch_idx + 1} 摘要:")
            print(f"  - 预测新事实平均置信度: {new_mu.mean().item():.4f}")
            print(f"  - 全局平均变动幅度: {change_mean:.6f}")
            print(f"  - 局部最大变动幅度: {change_max:.4f}")
            print(f"  - 达到阈值受影响的旧事实数: {affected_count}")

    print("\n=== 所有增量更新任务执行完毕，执行最终全面评估 ===")

    # 6. 构建 Base + Inc 合并图，用于最终评估
    inc_train_facts = dataset.inc_train
    if inc_train_facts:
        inc_h = torch.tensor(
            [f[0] for f in inc_train_facts], dtype=torch.long, device=args.device
        )
        inc_r = torch.tensor(
            [f[1] for f in inc_train_facts], dtype=torch.long, device=args.device
        )
        inc_t = torch.tensor(
            [f[2] for f in inc_train_facts], dtype=torch.long, device=args.device
        )
        inc_c = torch.tensor(
            [
                dataset.belief_state.get((f[0], f[1], f[2]), 0.5)
                for f in inc_train_facts
            ],
            dtype=torch.float,
            device=args.device,
        )
        eval_edge_index = torch.cat(
            [edge_index, torch.stack([inc_h, inc_t])], dim=1
        )
        eval_edge_type = torch.cat([edge_type, inc_r])
        eval_edge_conf = torch.cat([edge_conf, inc_c])
    else:
        eval_edge_index = edge_index
        eval_edge_type  = edge_type
        eval_edge_conf  = edge_conf

    model.eval()
    with torch.no_grad():
        final_z = model(eval_edge_index, eval_edge_type, eval_edge_conf)

    # 评估 1：Inc Test（新知识掌握）
    inc_test_metrics = evaluate_model(
        model, dataset.inc_test, z=final_z, device=args.device
    )
    Logger.print_metrics("Post-update Model on Inc Test", inc_test_metrics)

    # 评估 2：Base Test（旧知识记忆 / 灾难性遗忘）
    base_test_metrics = evaluate_model(
        model, dataset.base_test, z=final_z, device=args.device
    )
    Logger.print_metrics("Post-update Model on Base Test", base_test_metrics)

    # 评估 3：Combined Test（综合）
    print("\n[Joint Evaluation] 进行 Base 和 Inc 联合测试集评估...")
    if isinstance(dataset.base_test, list):
        combined_test = dataset.base_test + dataset.inc_test
    else:
        import numpy as np
        combined_test = np.concatenate([dataset.base_test, dataset.inc_test], axis=0)

    combined_test_metrics = evaluate_model(
        model, combined_test, z=final_z, device=args.device
    )
    Logger.print_metrics(
        "Post-update Model on Combined Test (Base + Inc)", combined_test_metrics
    )

    # 7. 保存 belief state
    os.makedirs("checkpoints", exist_ok=True)
    belief_path = f"checkpoints/final_belief_state_{dataset_name}_{baseline_name}.pt"
    torch.save(dataset.belief_state, belief_path)
    print(f">>> 全局 Belief 状态已保存至 {belief_path}")

    return {
        "inc_test":      inc_test_metrics,
        "base_test":     base_test_metrics,
        "combined_test": combined_test_metrics,
    }


def _build_arg_parser():
    """在 config.py 的 get_args 基础上追加 --baseline 参数。"""
    # 先让 config.py 完成解析，再手动添加 --baseline
    # 由于 argparse 不支持在已有 parser 上追加，这里采用两段式解析
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--baseline",
        type=str,
        default="ewc",
        choices=list(BASELINE_REGISTRY.keys()),
        help="选择基线模型: " + ", ".join(BASELINE_REGISTRY.keys()),
    )
    return pre_parser


if __name__ == "__main__":
    # 先解析 --baseline，再交由 config.get_args 解析其余参数
    import sys

    pre_parser = _build_arg_parser()
    pre_args, remaining = pre_parser.parse_known_args()

    # 将 remaining 传给 get_args，避免 --baseline 造成 argparse 冲突
    args = get_args(argv=remaining)
    args.baseline = pre_args.baseline

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_baseline(args)

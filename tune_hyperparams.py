"""
tune_hyperparams.py — Hyperparameter Tuning for UKG-Updater

Supports three tuning stages:
  --stage base      : Tune base GNN training hyperparameters
  --stage inc       : Tune incremental updating core parameters (requires pre-trained base)
  --stage finetune  : Tune propagation & fine-tuning parameters (requires pre-trained base)
  --stage all       : Run all three stages sequentially

Supports two search strategies:
  --mode optuna     : Bayesian optimisation via Optuna TPE sampler with MedianPruner
  --mode grid       : Exhaustive grid search over a pre-defined parameter grid

Usage examples:
  python tune_hyperparams.py --stage base --mode optuna --n_trials 50
  python tune_hyperparams.py --stage inc  --mode grid   --base_weight_path checkpoints/base_model_nl27k_ind.pth
  python tune_hyperparams.py --stage all  --mode optuna --n_trials 30 --n_seeds 3
"""

import argparse
import contextlib
import copy
import io
import itertools
import json
import os
import shutil
import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# Optional Optuna import
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from config import get_args as _get_config_args
from train_base import train_base
from run_incremental import run_incremental_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_default_model_args():
    """Return an argparse.Namespace with all config.py defaults (no sys.argv)."""
    args = _get_config_args(argv=[])
    args.data_dir = _sanitize_data_dir(args.data_dir)
    return args


def _sanitize_data_dir(data_dir):
    """Strip trailing commas, slashes, and whitespace from a data_dir path."""
    if data_dir is None:
        return data_dir
    return data_dir.strip().rstrip(",/\\")


def _suppress_output():
    """Context manager that swallows stdout/stderr when tuning_mode is active."""
    return contextlib.redirect_stdout(io.StringIO())


class _SuppressAll:
    """Context manager that redirects both stdout and stderr to /dev/null."""

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._devnull.close()


LARGE_PENALTY = 1e6


def _set_seed(args, seed):
    args.seed = seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_copy_checkpoint(src_path, dest_path):
    """Copy a checkpoint file from src to dest, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.copy2(src_path, dest_path)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def _run_base_trial(base_args, seeds):
    """Train base model and return average Test MSE across seeds."""
    mse_list = []
    for seed in seeds:
        args = copy.deepcopy(base_args)
        args.data_dir = _sanitize_data_dir(args.data_dir)
        _set_seed(args, seed)
        args.tuning_mode = True
        try:
            with _SuppressAll():
                metrics = train_base(args)
        except Exception as e:
            print(f"  [WARN] Base trial failed: {e}")
            return LARGE_PENALTY
        if metrics is None:
            return LARGE_PENALTY
        mse = metrics.get("MSE", LARGE_PENALTY)
        if not _is_finite(mse):
            return LARGE_PENALTY
        mse_list.append(mse)
    return sum(mse_list) / len(mse_list) if mse_list else LARGE_PENALTY


def _run_inc_trial(base_args, seeds, base_weight_path=None):
    """
    Run incremental update and return weighted objective:
        objective = Inc_Test_MSE + 0.3 * Base_Test_MSE
    """
    obj_list = []
    for seed in seeds:
        args = copy.deepcopy(base_args)
        args.data_dir = _sanitize_data_dir(args.data_dir)
        _set_seed(args, seed)
        args.tuning_mode = True

        # Ensure checkpoint exists at the expected path
        dataset_name = os.path.basename(os.path.normpath(args.data_dir))
        expected_ckpt = f"checkpoints/base_model_{dataset_name}.pth"
        if base_weight_path and os.path.normpath(base_weight_path) != os.path.normpath(expected_ckpt):
            try:
                _safe_copy_checkpoint(base_weight_path, expected_ckpt)
            except Exception as e:
                print(f"  [WARN] Failed to copy checkpoint: {e}")
                return LARGE_PENALTY

        try:
            with _SuppressAll():
                result = run_incremental_update(args)
        except Exception as e:
            print(f"  [WARN] Trial failed: {e}")
            return LARGE_PENALTY
        if result is None:
            return LARGE_PENALTY
        inc_mse = result.get("inc_test", {}).get("MSE", LARGE_PENALTY)
        base_mse = result.get("base_test", {}).get("MSE", LARGE_PENALTY)
        if not (_is_finite(inc_mse) and _is_finite(base_mse)):
            return LARGE_PENALTY
        obj_list.append(inc_mse + 0.3 * base_mse)
    return sum(obj_list) / len(obj_list) if obj_list else LARGE_PENALTY


def _is_finite(value):
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

STAGE_BASE_GRID = {
    "base_lr": [0.0005, 0.001, 0.005],
    "dropout_rate": [0.1, 0.3, 0.5],
    "alpha_cl": [0.1, 0.5, 1.0],
    "num_layers": [1, 2, 3],
    "base_weight_decay": [1e-5, 1e-4, 1e-3],
}

STAGE_INC_GRID = {
    "inc_lr": [1e-4, 0.001, 0.005, 0.01],
    "lambda_reg": [0.001, 0.01, 0.1, 0.5],
    "gamma": [0.5, 0.7, 0.8, 0.95],
    "influence_threshold": [0.005, 0.01, 0.05, 0.1],
    "refine_steps": [1, 3, 5],
}

STAGE_FINETUNE_GRID = {
    "alpha_base": [0.5, 1.0, 2.0],
    "mlp_anchor_coeff": [0.001, 0.01, 0.1],
    "finetune_steps": [3, 5, 10],
    "propagation_hops": [1, 2, 3],
}


def _suggest_base_params(trial):
    return {
        "base_lr": trial.suggest_float("base_lr", 0.0005, 0.005),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "alpha_cl": trial.suggest_float("alpha_cl", 0.1, 1.0),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "base_weight_decay": trial.suggest_float("base_weight_decay", 1e-5, 1e-3, log=True),
    }


def _suggest_inc_params(trial):
    return {
        "inc_lr": trial.suggest_float("inc_lr", 1e-4, 0.05, log=True),
        "lambda_reg": trial.suggest_float("lambda_reg", 0.001, 0.5, log=True),
        "gamma": trial.suggest_float("gamma", 0.5, 0.99),
        "influence_threshold": trial.suggest_float("influence_threshold", 0.005, 0.1, log=True),
        "refine_steps": trial.suggest_categorical("refine_steps", [1, 2, 3, 5, 8]),
    }


def _suggest_finetune_params(trial):
    return {
        "alpha_base": trial.suggest_float("alpha_base", 0.1, 5.0),
        "mlp_anchor_coeff": trial.suggest_float("mlp_anchor_coeff", 0.001, 0.1, log=True),
        "finetune_steps": trial.suggest_categorical("finetune_steps", [2, 3, 5, 8, 10, 15]),
        "propagation_hops": trial.suggest_categorical("propagation_hops", [1, 2, 3]),
    }


# ---------------------------------------------------------------------------
# Grid search helper
# ---------------------------------------------------------------------------

def _grid_combinations(grid):
    """Yield all combinations from a dict-of-lists grid."""
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _run_grid_search(stage, tune_args, model_base_args, seeds):
    """Run exhaustive grid search for a given stage. Returns list of result dicts."""
    grid = {
        "base": STAGE_BASE_GRID,
        "inc": STAGE_INC_GRID,
        "finetune": STAGE_FINETUNE_GRID,
    }[stage]

    combos = list(_grid_combinations(grid))
    print(f"  Grid search: {len(combos)} combinations")

    results = []
    for i, combo in enumerate(combos):
        args = copy.deepcopy(model_base_args)
        for k, v in combo.items():
            setattr(args, k, v)

        if stage == "base":
            objective = _run_base_trial(args, seeds)
        else:
            objective = _run_inc_trial(args, seeds, tune_args.base_weight_path)

        row = dict(combo)
        row["objective"] = objective
        results.append(row)

        print(f"  [{i+1}/{len(combos)}] {combo} → objective={objective:.6f}")

    return results


# ---------------------------------------------------------------------------
# Optuna tuning helpers
# ---------------------------------------------------------------------------

def _run_optuna(stage, tune_args, model_base_args, seeds):
    """Run Optuna TPE search for a given stage. Returns list of result dicts."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        )

    suggest_fn = {
        "base": _suggest_base_params,
        "inc": _suggest_inc_params,
        "finetune": _suggest_finetune_params,
    }[stage]

    results = []

    def objective(trial):
        params = suggest_fn(trial)
        args = copy.deepcopy(model_base_args)
        for k, v in params.items():
            setattr(args, k, v)

        if stage == "base":
            value = _run_base_trial(args, seeds)
        else:
            value = _run_inc_trial(args, seeds, tune_args.base_weight_path)

        results.append({**params, "objective": value})
        return value

    sampler = TPESampler(seed=model_base_args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    # Suppress Optuna's own verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=tune_args.n_trials, show_progress_bar=False)
    return results, study


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def _save_results(results, output_dir, stage, mode):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"results_{stage}_{mode}.csv")

    if not results:
        return

    import csv
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {csv_path}")


def _save_best_params(best_params, output_dir, stage):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"best_params_{stage}.json")
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best params saved to: {json_path}")


def _print_summary(results, stage):
    """Print a summary table for completed trials."""
    if not results:
        print("  No results to summarise.")
        return

    valid = [r for r in results if _is_finite(r["objective"])]
    if not valid:
        print("  All trials returned invalid (NaN/Inf) objectives.")
        return

    best = min(valid, key=lambda r: r["objective"])
    print(f"\n{'='*60}")
    print(f"  Stage '{stage}' summary — {len(valid)}/{len(results)} valid trials")
    print(f"  Best objective : {best['objective']:.6f}")
    param_keys = [k for k in best.keys() if k != "objective"]
    print("  Best params    :")
    for k in param_keys:
        print(f"    {k}: {best[k]}")
    print(f"{'='*60}\n")


def _best_params_from_results(results):
    valid = [r for r in results if _is_finite(r["objective"])]
    if not valid:
        return {}
    best = min(valid, key=lambda r: r["objective"])
    return {k: v for k, v in best.items() if k != "objective"}


# ---------------------------------------------------------------------------
# Per-stage runners
# ---------------------------------------------------------------------------

def run_stage(stage, tune_args, model_base_args):
    seeds = [model_base_args.seed + i for i in range(max(tune_args.n_seeds, 1))]

    print(f"\n{'#'*60}")
    print(f"  Tuning stage: {stage.upper()}  |  mode: {tune_args.mode}")
    print(f"  Seeds: {seeds}  |  data_dir: {model_base_args.data_dir}")
    print(f"{'#'*60}")

    if tune_args.mode == "optuna":
        results_list, study = _run_optuna(stage, tune_args, model_base_args, seeds)
    else:  # grid
        results_list = _run_grid_search(stage, tune_args, model_base_args, seeds)

    _print_summary(results_list, stage)
    _save_results(results_list, tune_args.output_dir, stage, tune_args.mode)

    best = _best_params_from_results(results_list)
    _save_best_params(best, tune_args.output_dir, stage)

    return best


# ---------------------------------------------------------------------------
# Argument parser for tuning script
# ---------------------------------------------------------------------------

def _parse_tune_args():
    parser = argparse.ArgumentParser(
        description="UKG-Updater Hyperparameter Tuning Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["base", "inc", "finetune", "all"],
        default="all",
        help="Which tuning stage(s) to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["optuna", "grid"],
        default="optuna",
        help="Search strategy: Bayesian optimisation (optuna) or exhaustive grid search (grid)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials per stage (ignored for grid mode)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset directory (defaults to config.py default: datasets/nl27k_ind)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of random seeds to average over per trial (use 3 for final evaluation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tuning_results",
        help="Directory where CSV and JSON results are saved",
    )
    parser.add_argument(
        "--base_weight_path",
        type=str,
        default=None,
        help="Path to pre-trained base model checkpoint (required for stages inc/finetune when not running stage 'all')",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    tune_args = _parse_tune_args()

    # Build base model args from config defaults (no sys.argv)
    model_base_args = _get_default_model_args()

    # Override data_dir if provided, then sanitize
    if tune_args.data_dir is not None:
        model_base_args.data_dir = tune_args.data_dir
    model_base_args.data_dir = _sanitize_data_dir(model_base_args.data_dir)

    # Check Optuna availability early
    if tune_args.mode == "optuna" and not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is not installed. Install it with: pip install optuna")
        sys.exit(1)

    stages_to_run = ["base", "inc", "finetune"] if tune_args.stage == "all" else [tune_args.stage]

    # Track best params from each stage to carry forward
    accumulated_best = {}

    for stage in stages_to_run:
        # For inc/finetune we need a base checkpoint
        if stage in ("inc", "finetune"):
            dataset_name = os.path.basename(os.path.normpath(model_base_args.data_dir))
            expected_ckpt = f"checkpoints/base_model_{dataset_name}.pth"

            # Use explicitly provided base_weight_path if given
            src_path = tune_args.base_weight_path

            # When running "all" stages, the base stage above will have saved a checkpoint
            # at the expected location already.  Only copy when an explicit path is given
            # and it differs from the expected location.
            if src_path and os.path.normpath(src_path) != os.path.normpath(expected_ckpt):
                if not os.path.exists(src_path):
                    print(f"ERROR: --base_weight_path '{src_path}' does not exist.")
                    sys.exit(1)
                _safe_copy_checkpoint(src_path, expected_ckpt)
                print(f"  Copied base checkpoint: {src_path} → {expected_ckpt}")
            elif not os.path.exists(expected_ckpt):
                print(
                    f"ERROR: Pre-trained base checkpoint not found at '{expected_ckpt}'.\n"
                    "  Please either:\n"
                    "    (a) Run stage 'base' first, or\n"
                    "    (b) Provide --base_weight_path <path>"
                )
                sys.exit(1)

        # Apply accumulated best params from previous stages to the model base args
        for k, v in accumulated_best.items():
            setattr(model_base_args, k, v)

        best_params = run_stage(stage, tune_args, model_base_args)
        accumulated_best.update(best_params)

    # Save a combined best-params file covering all stages run
    if accumulated_best:
        combined_path = os.path.join(tune_args.output_dir, "best_params_combined.json")
        os.makedirs(tune_args.output_dir, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(accumulated_best, f, indent=2)
        print(f"\nAll best params saved to: {combined_path}")

    print("\nTuning complete.")


if __name__ == "__main__":
    main()

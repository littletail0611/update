```markdown
# Uncertain Knowledge Graph Updater (UKG-Updater)

This repository contains the official PyTorch implementation of the dynamic confidence updating framework for Uncertain Knowledge Graphs (UKGs). The framework elegantly combines Heteroscedastic Graph Neural Networks, Expectation-Maximization (EM), Causal Influence Graphs, and Bayesian Belief Filtering to update knowledge confidences incrementally without retraining the entire graph.

## 📂 Directory Structure

```text
UKG_Updater/
├── data/
│   └── CN15K/              # UKG dataset (must contain 'base' and 'inc' folders)
│       ├── base/           # train.txt, valid.txt, test.txt (Initial graph)
│       └── inc/            # train.txt, valid.txt, test.txt (Incremental facts)
├── dataset.py              # Data loading, ID mapping, and global belief state management
├── model.py                # Confidence-aware GNN and Heteroscedastic dual-branch predictors
├── updater.py              # Core logic: Local EM, Causal Graph, and Bayesian Filtering
├── utils.py                # Evaluation metrics (MSE, MAE, RMSE) and loggers
├── train_base.py           # Script for Phase 1: Offline Base Model Pre-training
└── run_incremental.py      # Script for Phase 2: Online Incremental Belief Updating

```

## ⚙️ Environment Requirements

* Python >= 3.8
* PyTorch >= 1.12.0
* PyTorch Geometric (PyG) >= 2.1.0
* scikit-learn, numpy

Install dependencies via:

```bash
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
pip install torch_geometric scikit-learn numpy

```

## 📊 Data Format

The framework expects datasets to be split into `base` (offline pre-training) and `inc` (online updates) folders.
Each `.txt` file should contain tab-separated quadruples:

```text
head_entity    relation    tail_entity    confidence
/m/012s1       /r/born_in  /m/0245v       0.85
...

```

*Note: During the `inc` phase, confidences for new facts are assumed unobserved during inference and are only used for evaluation.*

## 🚀 How to Run

### Phase 1: Offline Base Pre-training

Train the Confidence-Aware GNN on the base graph. The model learns semantic representations and the heteroscedastic variance (epistemic uncertainty).

```bash
python train_base.py

```

* **Outputs**: `checkpoints/base_model.pth`
* **Monitors**: Heteroscedastic Gaussian Log-likelihood ($L_{conf}$).

### Phase 2: Online Incremental Updating

Load the pre-trained model and stream the incremental new facts. The script performs:

1. **Local EM Inference**: Generates heteroscedastic pseudo-labels for new facts.
2. **Causal Influence Graph**: Computes counterfactual influence $I(\tau)$ on old facts.
3. **Bayesian Belief Filtering**: Updates the global belief state using convex combinations.
4. **Local Refinement**: Fine-tunes affected entity embeddings.

```bash
python run_incremental.py

```

* **Outputs**: Evaluates MSE/MAE for the new facts and measures the correction accuracy of the global belief state for old facts. Saves `checkpoints/final_belief_state.pt`.

## 🔍 Hyperparameter Tuning

The `tune_hyperparams.py` script provides automated hyperparameter search using either **Optuna Bayesian optimisation** or **exhaustive Grid Search**, organised into three sequential tuning stages.

### Installation

```bash
pip install optuna   # required only for --mode optuna (Bayesian optimisation)
                     # grid search mode works without any additional dependencies
```

### 3-Stage Tuning Strategy

| Stage | Flag | Parameters tuned | Objective |
|-------|------|-----------------|-----------|
| **Base** | `--stage base` | `base_lr`, `dropout_rate`, `alpha_cl`, `num_layers`, `base_weight_decay` | Minimise Base Test MSE |
| **Inc** | `--stage inc` | `inc_lr`, `lambda_reg`, `gamma`, `influence_threshold`, `refine_steps` | `Inc_MSE + 0.3 × Base_MSE` |
| **Finetune** | `--stage finetune` | `alpha_base`, `mlp_anchor_coeff`, `finetune_steps`, `propagation_hops` | `Inc_MSE + 0.3 × Base_MSE` |

### Example Commands

**Run all stages sequentially with Optuna (50 trials each):**

```bash
python tune_hyperparams.py --stage all --mode optuna --n_trials 50
```

**Tune only the base model (grid search):**

```bash
python tune_hyperparams.py --stage base --mode grid
```

**Tune inc parameters using a pre-trained base checkpoint:**

```bash
python tune_hyperparams.py --stage inc --mode optuna --n_trials 30 \
    --base_weight_path checkpoints/base_model_nl27k_ind.pth
```

**Tune finetune parameters with 3 seeds for robustness:**

```bash
python tune_hyperparams.py --stage finetune --mode optuna --n_trials 30 \
    --n_seeds 3 --base_weight_path checkpoints/base_model_nl27k_ind.pth
```

**Use a different dataset:**

```bash
python tune_hyperparams.py --stage all --mode optuna --data_dir datasets/cn15k_ind
```

### Interpreting Results

All results are written to `tuning_results/` (configurable via `--output_dir`):

* `results_{stage}_{mode}.csv` — all trials with parameter values and objective scores
* `best_params_{stage}.json` — best hyperparameters found for that stage
* `best_params_combined.json` — merged best parameters across all stages run

After tuning, apply the best parameters by passing them as command-line arguments:

```bash
python train_base.py --base_lr 0.002 --dropout_rate 0.2 --alpha_cl 0.7
python run_incremental.py --inc_lr 0.003 --lambda_reg 0.05 --gamma 0.85
```

### CLI Reference

```
--stage {base,inc,finetune,all}   Which stage(s) to tune (default: all)
--mode  {optuna,grid}             Search strategy (default: optuna)
--n_trials N                      Optuna trials per stage (default: 50)
--data_dir PATH                   Dataset directory (default: datasets/nl27k_ind)
--n_seeds N                       Seeds to average per trial (default: 1)
--output_dir PATH                 Where to save results (default: tuning_results/)
--base_weight_path PATH           Pre-trained base checkpoint for inc/finetune stages
```

## 🧠 Core Methodology Components

* **Heteroscedastic Confidence Prediction (`model.py`)**: Outputs both $\mu_\tau$ and $\sigma_\tau^2$ to dynamically scale learning weights.
* **Confidence-Aware Message Passing (`model.py`)**: Uses prior confidences to regulate semantic attention scores during GNN aggregation.
* **Unified Confidence Updater (`updater.py`)**: Solves the unobservable old-fact confidence evolution as a belief revision probability task $P(\mathbf{C}_{new}, \mathbf{C}_{update} | \mathcal{G}, \mathcal{B})$.
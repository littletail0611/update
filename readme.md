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

## 🧠 Core Methodology Components

* **Heteroscedastic Confidence Prediction (`model.py`)**: Outputs both $\mu_\tau$ and $\sigma_\tau^2$ to dynamically scale learning weights.
* **Confidence-Aware Message Passing (`model.py`)**: Uses prior confidences to regulate semantic attention scores during GNN aggregation.
* **Unified Confidence Updater (`updater.py`)**: Solves the unobservable old-fact confidence evolution as a belief revision probability task $P(\mathbf{C}_{new}, \mathbf{C}_{update} | \mathcal{G}, \mathcal{B})$.
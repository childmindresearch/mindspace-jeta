# MindSpace-CLIP: Free-Text to PCA Psychological Component Metric Space

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-3.0%2B-ff6f00.svg)](https://www.sbert.net/)
[![Optuna](https://img.shields.io/badge/Optuna-4.0%2B-44A833.svg)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MindSpace-CLIP** is a complete, modular, and config-driven Python pipeline in **PyTorch** that implements a **CLIP-style Contrastive Projection Model** to align unstructured free-text journal entries (~150–300 words) with a multi-dimensional PCA psychological component score space.

---

## 📌 Architectural Overview

The system maps both **high-dimensional text embeddings** (e.g., 384D from `all-MiniLM-L6-v2` or 1024D from `bge-large-en-v1.5`) and **low-dimensional PCA score vectors** (e.g., 5D survey scores: *Task, Intrusive, Emotion, etc.*) into a **shared 16-dimensional L2-normalized metric space** using symmetric InfoNCE / CLIP cross-entropy loss.

```
 ┌─────────────────────────┐               ┌─────────────────────────┐
 │   Journal Text Entry    │               │  5D Survey PCA Vector   │
 │   (150-300 words)       │               │ [Task, Intrusive, ...]  │
 └────────────┬────────────┘               └────────────┬────────────┘
              │                                         │
    SentenceTransformer                                 │
  (all-MiniLM-L6-v2, 256)                              │
              │ (384-D)                                 │ (5-D)
              ▼                                         ▼
┌───────────────────────────┐             ┌───────────────────────────┐
│    Text Projection Head   │             │   PCA Projection Head     │
│   Linear(384 -> 128)      │             │     Linear(5 -> 32)       │
│   BatchNorm1d(128)        │             │     BatchNorm1d(32)       │
│   ReLU()                  │             │     ReLU()                │
│   Dropout(p=0.3)          │             │     Linear(32 -> 16)      │
│   Linear(128 -> 16)       │             └─────────────┬─────────────┘
└─────────────┬─────────────┘                           │
              │                                         │
              ▼                                         ▼
      L2-Normalization                          L2-Normalization
              │                                         │
              └──────────────────┬──────────────────────┘
                                 ▼
                     16-D Shared Metric Space
                  Symmetric InfoNCE / CLIP Loss
```

---

## 🚀 Key Features

- ⚙️ **Fully Config-Driven (`config.yaml`)**: Manage embedding models, text sequence lengths, PCA column headers, hidden layer dimensions, hyperparameters, and file output paths without modifying code.
- 🧬 **Dynamic Projection Heads**: Automatically scales input text dimensions (384D/1024D) and PCA component counts (5D, 8D, 10D).
- 🔍 **Direct Text-to-PCA Prediction**: Maps $1...N$ free-text entries directly into predicted 5D PCA component score matrices (`outputs/predicted_pca_scores.csv`).
- 🔄 **Cross-Modal Retrieval**: Query ranked journal entries matching arbitrary target 5D PCA psychological profiles.
- 🎯 **Optuna Fine-Tuning & Archiving (`tune.py`)**: Automatic TPE hyperparameter optimization. Prior configurations are automatically archived to `config_archive/config_YYYYMMDD_HHMMSS.yaml` before updating `config.yaml` in-place.
- 📊 **Detailed Training Logs & Diagnostics**: Generates epoch-by-epoch CSV metrics (`training_metrics.csv`), summary JSON (`training_summary.json`), and 4-panel diagnostic plots (`loss_curves.png`).
- 📈 **Held-Out Test Evaluation (`evaluate.py`)**: Computes Top-1/Top-5 accuracy, Mean Reciprocal Rank (MRR), $R^2$ variance score, MAE, RMSE, and separation margin.

---

## 📂 Project Structure

```
CLIP/
├── config.yaml              # Unified configuration file (CSV paths, column headers, model hparams)
├── config_archive/          # Archive folder storing timestamped prior configurations
│   └── config_20260827_112446.yaml
├── train.py                 # Model training & validation orchestrator
├── predict.py               # Direct text-to-PCA prediction & cross-modal retrieval
├── evaluate.py              # Held-out test performance evaluation & metrics report
├── tune.py                  # Optuna hyperparameter optimization & fine-tuning orchestrator
├── data/                    # Dedicated data directory for input .csv files
│   ├── train_data.csv       # Training CSV dataset
│   └── eval_data.csv        # Evaluation CSV dataset
├── utils/                   # Standalone data analysis utilities
│   ├── __init__.py
│   └── analyze_token_length.py # Tokenizer scanner & 95% coverage recommendation tool
├── outputs/                 # Directory for model checkpoints and output embeddings
│   ├── contrastive_model.pt # Trained model checkpoint
│   ├── predicted_pca_scores.csv   # Text-to-5D PCA predictions
│   └── logs/                # Training metrics CSV, summary JSON, and 4-panel diagnostic plots
└── src/                     # Core package library
    ├── __init__.py
    ├── config.py            # Dataclass loader, validator, exporter & timestamp archiver
    ├── models.py            # PyTorch ContrastiveProjectionModel (Text Head, PCA Head, Temperature)
    ├── dataset.py           # PyTorch Dataset for paired mini-batch loading
    ├── utils.py             # SentenceTransformer wrapper, CSV parser, synthetic generator
    ├── trainer.py           # Symmetric InfoNCE Loss, AdamW optimizer, metrics tracking & plotting
    └── inference.py         # Shared embedding projection & 5D PCA mapping engine
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Configuration (`config.yaml`)

Manage all settings through `config.yaml`:

```yaml
dataset:
  train_csv_path: "./data/train_data.csv"
  inference_csv_path: "./data/eval_data.csv"
  text_column: "journal_entry"
  pca_columns:
    - "PCA_1"
    - "PCA_2"
    - "PCA_3"
    - "PCA_4"
    - "PCA_5"
  sample_query_pca:
    - 1.5
    - -1.0
    - 0.5
    - 0.0
    - 0.5

text_encoder:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  max_seq_length: 256
  text_input_dim: null  # Auto-detected

model_architecture:
  shared_dim: 16
  text_head_hidden_dims:
    - 128
  text_head_dropout: 0.3
  pca_head_hidden_dims:
    - 32
  initial_temperature: 0.07

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 25
  train_split: 0.8
  seed: 42

optuna:
  n_trials: 25
  timeout: null
  best_config_path: "./config.yaml"
  archive_dir: "./config_archive"
```

---

## 💻 Workflow Guide

### 1. Dataset Length & Token Scanner
Scans your input CSV file, tokenizes all free-text entries, and calculates the recommended `max_seq_length` to cover 95% (or any custom percentile) of entries without truncation:
```bash
python utils/analyze_token_length.py --percentile 95.0 --update_config
```

### 2. Optuna Hyperparameter Optimization & Fine-Tuning
Searches over learning rates, weight decay, hidden dimensions, dropout, temperature, and batch sizes. Automatically archives the current `config.yaml` to `config_archive/config_YYYYMMDD_HHMMSS.yaml` and updates `config.yaml` in-place:
```bash
python tune.py --n_trials 25
```

### 3. Training the Model
Runs mini-batch training with an 80/20 train/validation split using `config.yaml`, logs metrics per epoch to `outputs/logs/`, and saves model checkpoints:
```bash
python train.py
```

### 4. Direct Text-to-PCA Prediction & Cross-Modal Retrieval
Projects test texts to 16D shared space, exports estimated 5D PCA component scores to `outputs/predicted_pca_scores.csv`, and runs sample cross-modal retrieval queries:
```bash
python predict.py
```

### 5. Held-Out Test Performance Evaluation
Runs a rigorous accuracy evaluation on a held-out test dataset, computing Top-1/Top-5 retrieval accuracy, MRR, $R^2$ scores, MAE, and separation margins:
```bash
python evaluate.py --test_csv ./data/eval_data.csv
```

---

## 🐍 Python API Usage Example

```python
from src.inference import CLIPPCAPipeline
import pandas as pd

# 1. Load trained pipeline checkpoint
pipeline = CLIPPCAPipeline.load_from_checkpoint('./outputs/contrastive_model.pt')

# 2. Load reference dataset
ref_df = pd.read_csv('./data/train_data.csv')
ref_pca = ref_df[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']].to_numpy()

# 3. Predict 5D PCA Component Scores for 1...N free-text entries
journal_entries = [
    "Today was productive. Finished work early and organized my entire workspace.",
    "Feeling overwhelmed by upcoming deadlines. Intrusive thoughts repeating constantly."
]

predicted_5d_pca = pipeline.predict_pca_components(
    text_list=journal_entries,
    reference_pca_matrix=ref_pca
)

print("Predicted 5D PCA Matrix Shape:", predicted_5d_pca.shape)
print(predicted_5d_pca)

# 4. Cross-Modal Retrieval Query (5D Target PCA Profile -> Ranked Text Entries)
query_pca_profile = [1.5, -1.0, 0.5, 0.0, 0.5]
results = pipeline.cross_modal_retrieval(
    query_pca_vector=query_pca_profile,
    text_database=journal_entries,
    top_k=2
)

for res in results:
    print(f"Rank #{res['rank']} | Sim: {res['similarity_score']:.4f} | Entry: {res['text']}")
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

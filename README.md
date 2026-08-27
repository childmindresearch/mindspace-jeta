# MindSpace-CLIP: Free-Text to PCA Psychological Component Metric Space

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-3.0%2B-ff6f00.svg)](https://www.sbert.net/)
[![Optuna](https://img.shields.io/badge/Optuna-4.0%2B-44A833.svg)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MindSpace-CLIP** is a Python pipeline in **PyTorch** that implements a **CLIP-style Contrastive Projection Model** to align unstructured free-text responses with a multi-dimensional PCA psychological component score space.

---

## 📌 Architectural Overview

The system maps **text embeddings** (extracted via `sentence-transformers`) and **PCA score vectors** into a **shared L2-normalized metric space** using symmetric InfoNCE cross-entropy loss.

```
 ┌─────────────────────────┐               ┌─────────────────────────┐
 │   Free-Text Response    │               │ 5D Survey PCA Vector    │
 └────────────┬────────────┘               └────────────┬────────────┘
              │                                         │
    SentenceTransformer                                 │
 (all-MiniLM-L6-v2, 256)                               │
              │ (384D)                                  │ (5D)
              ▼                                         ▼
┌───────────────────────────┐             ┌───────────────────────────┐
│   Text Projection Head    │             │   PCA Projection Head     │
│  Linear(384 -> 128)       │             │    Linear(5 -> 32)        │
│  BatchNorm1d(128)         │             │    BatchNorm1d(32)        │
│  ReLU()                   │             │    ReLU()                 │
│  Dropout(p=0.3)           │             │    Linear(32 -> 16)       │
│  Linear(128 -> 16)        │             └─────────────┬─────────────┘
└─────────────┬─────────────┘                           │
              │                                         │
              ▼                                         ▼
      L2-Normalization                          L2-Normalization
              │                                         │
              └──────────────────┬──────────────────────┘
                                 ▼
                     16D Shared Metric Space
                  Symmetric InfoNCE / CLIP Loss
```

---

## 🚀 Key Features

- **Config-Driven (`config.yaml`)**: Set text model, sequence length, PCA column headers, layer dimensions, training parameters, and file output paths.
- **Direct Text-to-PCA Estimation**: Predicts PCA component scores for text entries via similarity interpolation over reference exemplars (`outputs/predicted_pca_scores.csv`).
- **Cross-Modal Retrieval**: Ranks text entries matching a target PCA profile vector.
- **Optuna Tuning (`tune.py`)**: Optimizes learning rate, weight decay, layer dimensions, dropout, and batch size. Automatically archives prior `config.yaml` to `config_archive/` before updating in-place.
- **Metrics & Diagnostic Plotting**: Logs epoch metrics (`training_metrics.csv`), summary JSON (`training_summary.json`), and diagnostic loss curves (`loss_curves.png`).
- **Test Evaluation (`evaluate.py`)**: Computes Top-1/Top-5 retrieval accuracy, Mean Reciprocal Rank (MRR), $R^2$ scores, MAE, RMSE, and separation margins on test data.

---

## 📂 Project Structure

```
mindspace-clip/
├── config.yaml              # Pipeline configuration file
├── config_archive/          # Timestamped archived configuration files
├── train.py                 # Training and validation script
├── predict.py               # Text-to-PCA prediction & cross-modal retrieval script
├── evaluate.py              # Test set evaluation script
├── tune.py                  # Optuna hyperparameter optimization script
├── data/                    # CSV datasets (mdes_train.csv, mdes_test.csv)
├── utils/                   # Utilities (analyze_token_length.py)
├── outputs/                 # Checkpoints, predicted CSVs, and logs
└── src/                     # Core package library
    ├── config.py            # Dataclass loader, validator, exporter & archiver
    ├── models.py            # TextProjectionHead, PCAProjectionHead, ContrastiveProjectionModel
    ├── dataset.py           # JournalPCADataset PyTorch Dataset
    ├── utils.py             # TextEncoderWrapper, load_csv_dataset, generate_synthetic_csv
    ├── trainer.py           # SymmetricCLIPLoss, Trainer class, diagnostic plotting
    └── inference.py         # CLIPPCAPipeline inference engine
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

Example `config.yaml` schema:

```yaml
dataset:
  train_csv_path: "./data/mdes_train.csv"
  inference_csv_path: "./data/mdes_test.csv"
  text_column: "prompt_response"
  pca_columns:
    - "Detailed Task Focus"
    - "Intrusive Distraction"
    - "Episodic Social Cognition"
    - "Future Problem-Solving"
    - "Sensory Engagement"
  sample_query_pca:
    - 1.5
    - -1.0
    - 0.5
    - 0.0
    - 0.5

text_encoder:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  max_seq_length: 256
  text_input_dim: 384

model_architecture:
  shared_dim: 16
  text_head_hidden_dims:
    - 128
  text_head_dropout: 0.3
  pca_head_hidden_dims:
    - 32
  initial_temperature: 0.07

training:
  batch_size: 16
  learning_rate: 0.0005
  weight_decay: 0.05
  epochs: 25
  train_split: 0.8
  seed: 42

paths:
  output_dir: "./outputs"
  logs_dir: "./outputs/logs"
  model_checkpoint: "./outputs/contrastive_model.pt"
  embeddings_output: "./outputs/projected_embeddings.pt"

optuna:
  n_trials: 25
  timeout: null
  best_config_path: "./config.yaml"
  archive_dir: "./config_archive"
```

---

## 💻 Workflow Guide

### 1. Token Length Scanner
Analyzes text length distribution and recommends `max_seq_length`:
```bash
python utils/analyze_token_length.py --percentile 95.0 --update_config
```

### 2. Optuna Hyperparameter Optimization
Runs hyperparameter optimization over learning rate, weight decay, dimensions, and batch size:
```bash
python tune.py --n_trials 25
```

### 3. Model Training
Trains model using `config.yaml` parameters and logs metrics to `outputs/logs/`:
```bash
python train.py
```

### 4. Text-to-PCA Prediction & Retrieval
Generates PCA predictions and executes cross-modal retrieval query:
```bash
python predict.py
```

### 5. Test Evaluation
Evaluates trained checkpoint on test dataset:
```bash
python evaluate.py --test_csv ./data/mdes_test.csv
```

---

## 🐍 Python API Usage Example

```python
from src.inference import CLIPPCAPipeline
import pandas as pd

# 1. Load trained checkpoint
pipeline = CLIPPCAPipeline.load_from_checkpoint('./outputs/contrastive_model.pt')

# 2. Load reference dataset PCA scores
ref_df = pd.read_csv('./data/mdes_train.csv')
ref_pca = ref_df[['Detailed Task Focus', 'Intrusive Distraction', 'Episodic Social Cognition', 'Future Problem-Solving', 'Sensory Engagement']].to_numpy()

# 3. Estimate PCA scores for text entries
texts = [
    "Today was productive. Finished work early and organized my entire workspace.",
    "Feeling anxious about upcoming deadlines. Thoughts keep repeating."
]

predicted_pca = pipeline.predict_pca_components(
    text_list=texts,
    reference_pca_matrix=ref_pca
)

print("Predicted PCA Matrix Shape:", predicted_pca.shape)

# 4. Query text entries matching target PCA profile
query_profile = [1.5, -1.0, 0.5, 0.0, 0.5]
results = pipeline.cross_modal_retrieval(
    query_pca_vector=query_profile,
    text_database=texts,
    top_k=2
)

for res in results:
    print(f"Rank #{res['rank']} | Sim: {res['similarity_score']:.4f} | Text: {res['text']}")
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

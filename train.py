#!/usr/bin/env python3
"""
Training Orchestrator Script.
Loads YAML configuration, prepares training data (or generates synthetic data if missing),
encodes text entries, initializes ContrastiveProjectionModel, and runs model training & validation.
"""

import argparse
import os
import sys

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

from src.config import load_config, Config
from src.dataset import JournalPCADataset
from src.models import ContrastiveProjectionModel
from src.trainer import Trainer
from src.utils import TextEncoderWrapper, load_csv_dataset, generate_synthetic_csv


def resolve_config_path(config_arg: str) -> str:
    """If user did not specify a custom config file, automatically use config_best.yaml if it exists."""
    if config_arg != "config.yaml":
        return config_arg

    if os.path.exists("config_best.yaml"):
        print("[Config Resolution] Detected Optuna-tuned 'config_best.yaml'. Automatically using optimal configuration.")
        return "config_best.yaml"
    return "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Train CLIP-Style Contrastive Projection Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    print("=" * 70)
    print("      Contrastive Projection Model Training Pipeline")
    print("=" * 70)

    # 1. Load Configuration
    config = load_config(config_path)
    print(f"[1/5] Configuration loaded from '{config_path}'")
    print(f"      Text Model: {config.text_encoder.model_name}")
    print(f"      PCA Columns ({len(config.dataset.pca_columns)}): {config.dataset.pca_columns}")
    print(f"      Train CSV Path: {config.dataset.train_csv_path}")

    # 2. Check or Generate Dataset
    train_csv = config.dataset.train_csv_path
    if not os.path.exists(train_csv):
        print(f"[Warning] Training CSV '{train_csv}' not found. Generating synthetic dataset (N=500)...")
        generate_synthetic_csv(
            output_csv_path=train_csv,
            num_samples=500,
            text_column=config.dataset.text_column,
            pca_columns=config.dataset.pca_columns,
            seed=config.training.seed,
        )

    # 3. Read CSV Dataset
    print(f"[2/5] Reading CSV dataset from '{train_csv}'...")
    texts, pca_matrix = load_csv_dataset(
        csv_path=train_csv,
        text_column=config.dataset.text_column,
        pca_columns=config.dataset.pca_columns,
    )
    print(f"      Loaded {len(texts)} samples with PCA score matrix of shape {pca_matrix.shape}")

    # 4. Text Encoding via SentenceTransformer
    print(f"[3/5] Encoding text entries using '{config.text_encoder.model_name}' (max_seq_length={config.text_encoder.max_seq_length})...")
    text_encoder = TextEncoderWrapper(
        model_name=config.text_encoder.model_name,
        max_seq_length=config.text_encoder.max_seq_length,
    )

    text_embeddings = text_encoder.encode(texts, batch_size=config.training.batch_size, show_progress_bar=True)
    text_dim = text_encoder.embedding_dim
    print(f"      Extracted text embeddings shape: {text_embeddings.shape}")

    # Update config text_input_dim
    config.text_encoder.text_input_dim = text_dim

    # Create PyTorch Dataset
    dataset = JournalPCADataset(text_embeddings=text_embeddings, pca_scores=pca_matrix)

    # 5. Model Initialization & Training
    print(f"[4/5] Initializing ContrastiveProjectionModel (Text Head: {text_dim}->{config.model_architecture.text_head_hidden_dims}->16, PCA Head: {pca_matrix.shape[1]}->{config.model_architecture.pca_head_hidden_dims}->16)...")
    model = ContrastiveProjectionModel(
        text_input_dim=text_dim,
        pca_input_dim=pca_matrix.shape[1],
        shared_dim=config.model_architecture.shared_dim,
        text_head_hidden_dims=config.model_architecture.text_head_hidden_dims,
        text_head_dropout=config.model_architecture.text_head_dropout,
        pca_head_hidden_dims=config.model_architecture.pca_head_hidden_dims,
        initial_temperature=config.model_architecture.initial_temperature,
    )

    print(f"[5/5] Starting AdamW training for {config.training.epochs} epochs...")
    trainer = Trainer(model=model, config=config)
    history = trainer.train(dataset)

    print("\n" + "=" * 70)
    print(f" SUCCESS: Model trained & saved to '{config.paths.model_checkpoint}'")
    print(f" Logs & Diagnostics Directory: '{config.paths.logs_dir}'")
    print(f"   - Epoch Metrics CSV:  '{os.path.join(config.paths.logs_dir, 'training_metrics.csv')}'")
    print(f"   - Summary JSON:       '{os.path.join(config.paths.logs_dir, 'training_summary.json')}'")
    print(f"   - Loss & Acc Plots:   '{os.path.join(config.paths.logs_dir, 'loss_curves.png')}'")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hyperparameter Optimization & Fine-Tuning Orchestrator using Optuna.
Searches for optimal learning rates, weight decays, projection head dimensions,
shared metric space dimensions, dropout, and temperature values.
Saves the optimal configuration to config.yaml (archiving prior config to config_archive/).
"""

import argparse
import ast
import copy
import os
import sys
from typing import List

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

import numpy as np
import optuna
import torch

from src.config import load_config, save_config, archive_existing_config, Config
from src.dataset import JournalPCADataset
from src.models import ContrastiveProjectionModel
from src.trainer import Trainer
from src.utils import TextEncoderWrapper, load_csv_dataset, generate_synthetic_csv


import shutil

def objective(
    trial: optuna.Trial,
    base_config: Config,
    text_embeddings: np.ndarray,
    pca_matrix: np.ndarray,
    best_tracker: Dict[str, float],
) -> float:
    """Optuna objective evaluation function for a single trial."""
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    shared_dim = trial.suggest_categorical("shared_dim", [16, 32, 64])

    text_head_dims_str = trial.suggest_categorical("text_head_hidden_dims", ["[128]", "[256]", "[128, 64]"])
    text_head_hidden_dims: List[int] = ast.literal_eval(text_head_dims_str)

    text_head_dropout = trial.suggest_float("text_head_dropout", 0.1, 0.5, step=0.1)

    pca_head_dims_str = trial.suggest_categorical("pca_head_hidden_dims", ["[32]", "[64]", "[32, 16]"])
    pca_head_hidden_dims: List[int] = ast.literal_eval(pca_head_dims_str)

    initial_temperature = trial.suggest_float("initial_temperature", 0.01, 0.2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 2. Clone and Update Config for this Trial
    trial_config = copy.deepcopy(base_config)
    trial_config.training.learning_rate = learning_rate
    trial_config.training.weight_decay = weight_decay
    trial_config.training.batch_size = batch_size
    trial_config.model_architecture.shared_dim = shared_dim
    trial_config.model_architecture.text_head_hidden_dims = text_head_hidden_dims
    trial_config.model_architecture.text_head_dropout = text_head_dropout
    trial_config.model_architecture.pca_head_hidden_dims = pca_head_hidden_dims
    trial_config.model_architecture.initial_temperature = initial_temperature

    # Use shorter epochs per trial for fast search (e.g. 15 epochs)
    trial_epochs = min(15, base_config.training.epochs)
    trial_config.training.epochs = trial_epochs

    # Temporary checkpoint path for evaluating this trial
    temp_checkpoint_path = os.path.join(base_config.paths.output_dir, "temp_trial_best.pt")
    trial_config.paths.model_checkpoint = temp_checkpoint_path

    # 3. Create Dataset & Model
    dataset = JournalPCADataset(text_embeddings=text_embeddings, pca_scores=pca_matrix)

    model = ContrastiveProjectionModel(
        text_input_dim=text_embeddings.shape[1],
        pca_input_dim=pca_matrix.shape[1],
        shared_dim=shared_dim,
        text_head_hidden_dims=text_head_hidden_dims,
        text_head_dropout=text_head_dropout,
        pca_head_hidden_dims=pca_head_hidden_dims,
        initial_temperature=initial_temperature,
    )

    # Suppress verbose prints during trial execution
    trainer = Trainer(model=model, config=trial_config)

    # 4. Run Training Loop
    history = trainer.train(dataset)
    min_val_loss = float(min(history["val_loss"]))

    # 5. Overwrite global model checkpoint ONLY if this trial improves overall best loss
    if min_val_loss < best_tracker["loss"]:
        best_tracker["loss"] = min_val_loss
        final_checkpoint_path = base_config.paths.model_checkpoint
        os.makedirs(os.path.dirname(final_checkpoint_path) or ".", exist_ok=True)
        if os.path.exists(temp_checkpoint_path):
            shutil.copy2(temp_checkpoint_path, final_checkpoint_path)
            print(f" [New Global Best] Trial #{trial.number} set a new best val loss ({min_val_loss:.4f}). Saved to '{final_checkpoint_path}'")

    # Clean up temporary checkpoint
    if os.path.exists(temp_checkpoint_path):
        os.remove(temp_checkpoint_path)

    return min_val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Contrastive Projection Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--n_trials", type=int, default=None, help="Number of Optuna search trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for tuning")
    parser.add_argument("--save_best", type=str, default=None, help="Path to save updated configuration file")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate synthetic CSV dataset if training file is missing")
    args = parser.parse_args()

    print("=" * 70)
    print("      Optuna Hyperparameter Optimization Pipeline")
    print("=" * 70)

    # 1. Load Baseline Config
    base_config = load_config(args.config)
    n_trials = args.n_trials or base_config.optuna.n_trials
    timeout = args.timeout or base_config.optuna.timeout
    save_best_path = args.save_best or base_config.optuna.best_config_path

    print(f"[1/4] Loaded baseline config from '{args.config}'")
    print(f"      Search Trials: {n_trials}")
    print(f"      Target Config to Update: '{save_best_path}'")

    # 2. Check / Generate Dataset
    train_csv = base_config.dataset.train_csv_path
    if not os.path.exists(train_csv):
        if args.generate_synthetic:
            print(f"[Synthetic Data] Training CSV '{train_csv}' not found. Generating synthetic dataset (N=500)...")
            generate_synthetic_csv(
                output_csv_path=train_csv,
                text_column=base_config.dataset.text_column,
                pca_columns=base_config.dataset.pca_columns,
                num_samples=500,
                seed=base_config.training.seed,
            )
        else:
            print(f"[Dataset Error] Training dataset file not found at '{train_csv}'.")
            print("Please check 'dataset.train_csv_path' in 'config.yaml' or pass '--generate_synthetic' to create a test dataset.")
            sys.exit(1)

    # 3. Load & Pre-encode Dataset (Done ONCE to speed up trials!)
    print(f"[2/4] Reading dataset from '{train_csv}'...")
    texts, pca_matrix = load_csv_dataset(
        csv_path=train_csv,
        text_column=base_config.dataset.text_column,
        pca_columns=base_config.dataset.pca_columns,
    )

    print(f"[3/4] Pre-encoding {len(texts)} text entries using '{base_config.text_encoder.model_name}'...")
    text_encoder = TextEncoderWrapper(
        model_name=base_config.text_encoder.model_name,
        max_seq_length=base_config.text_encoder.max_seq_length,
    )
    text_embeddings = text_encoder.encode(texts, batch_size=64, show_progress_bar=False)
    base_config.text_encoder.text_input_dim = text_embeddings.shape[1]

    # 4. Initialize & Execute Optuna Study
    print(f"[4/4] Launching Optuna study over {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=base_config.training.seed),
    )

    # Global tracker to ensure contrastive_model.pt always holds the best weights across all trials
    best_tracker = {"loss": float("inf")}

    study.optimize(
        lambda trial: objective(trial, base_config, text_embeddings, pca_matrix, best_tracker),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    if not study.trials or study.best_value is None:
        print("[Error] No successful trials completed during Optuna optimization.")
        return

    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print(f" OPTUNA OPTIMIZATION COMPLETE (Trial #{best_trial.number})")
    print(f" Best Validation Loss: {best_trial.value:.4f}")
    print("-" * 70)
    print(" Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"   - {k}: {v}")

    # Build optimized Config dataclass
    best_config = copy.deepcopy(base_config)
    best_config.training.learning_rate = float(best_trial.params["learning_rate"])
    best_config.training.weight_decay = float(best_trial.params["weight_decay"])
    best_config.training.batch_size = int(best_trial.params["batch_size"])
    best_config.model_architecture.shared_dim = int(best_trial.params["shared_dim"])
    best_config.model_architecture.text_head_hidden_dims = ast.literal_eval(best_trial.params["text_head_hidden_dims"])
    best_config.model_architecture.text_head_dropout = float(best_trial.params["text_head_dropout"])
    best_config.model_architecture.pca_head_hidden_dims = ast.literal_eval(best_trial.params["pca_head_hidden_dims"])
    best_config.model_architecture.initial_temperature = float(best_trial.params["initial_temperature"])

    # Preserve configured output checkpoint paths
    best_config.paths.model_checkpoint = base_config.paths.model_checkpoint
    best_config.paths.embeddings_output = base_config.paths.embeddings_output

    # Archive existing config before overwriting with tuned parameters
    archive_existing_config(config_path=save_best_path, archive_dir=base_config.optuna.archive_dir)

    # Save tuned parameters in-place
    save_config(best_config, save_best_path)

    print(f"\n [Saved Best Weights] Target checkpoint '{base_config.paths.model_checkpoint}' holds overall best weights (Val Loss: {best_trial.value:.4f})")
    print(f" To inspect or run inference using these saved optimal weights:")
    print(f"   python src/inference.py")
    print("=" * 70)


if __name__ == "__main__":
    main()


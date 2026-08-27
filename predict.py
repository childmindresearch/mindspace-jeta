#!/usr/bin/env python3
"""
Direct Inference & Retrieval Orchestrator Script.
Loads trained checkpoint from config, loads a separate evaluation CSV dataset,
projects texts and PCA vectors into 16D shared space, saves output embeddings,
and executes cross-modal retrieval queries.
"""

import argparse
import os
import sys

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

import numpy as np
import torch

from src.config import load_config, Config
from src.inference import CLIPPCAPipeline
from src.utils import load_csv_dataset, generate_synthetic_csv


def resolve_config_path(config_arg: str) -> str:
    """If user did not specify a custom config file, automatically use config_best.yaml if it exists."""
    if config_arg != "config.yaml":
        return config_arg

    if os.path.exists("config_best.yaml"):
        print("[Config Resolution] Detected Optuna-tuned 'config_best.yaml'. Automatically using optimal configuration.")
        return "config_best.yaml"
    return "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="Run Direct Inference & Cross-Modal Retrieval")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--eval_csv", type=str, default=None, help="Override path to separate evaluation CSV dataset")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top retrieved entries to return")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    print("=" * 70)
    print("      Contrastive Projection Direct Inference & Retrieval")
    print("=" * 70)

    # 1. Load Configuration
    config = load_config(config_path)
    eval_csv_path = args.eval_csv or config.dataset.inference_csv_path

    # Check model checkpoint
    checkpoint_path = config.paths.model_checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Trained model checkpoint not found at '{checkpoint_path}'.")
        print("Please run `python train.py` first to train and save the model checkpoint.")
        sys.exit(1)

    # Check or generate separate evaluation CSV dataset
    if not os.path.exists(eval_csv_path):
        print(f"[Warning] Inference dataset CSV '{eval_csv_path}' not found. Generating synthetic evaluation dataset (N=100)...")
        generate_synthetic_csv(
            output_csv_path=eval_csv_path,
            num_samples=100,
            text_column=config.dataset.text_column,
            pca_columns=config.dataset.pca_columns,
            seed=123,
        )

    # 2. Load Evaluation Dataset
    print(f"[1/4] Loading evaluation CSV dataset from '{eval_csv_path}'...")
    eval_texts, eval_pca_matrix = load_csv_dataset(
        csv_path=eval_csv_path,
        text_column=config.dataset.text_column,
        pca_columns=config.dataset.pca_columns,
    )
    print(f"      Loaded {len(eval_texts)} evaluation entries.")

    # 3. Load Trained Pipeline Engine
    print(f"[2/4] Loading trained model pipeline from '{checkpoint_path}'...")
    pipeline = CLIPPCAPipeline.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)

    # 4. Direct Projection to 16D Shared Space
    print("[3/4] Projecting evaluation texts and PCA component matrices to 16D shared space...")
    text_shared_embeds = pipeline.predict_shared_embedding(eval_texts)
    pca_shared_embeds = pipeline.pca_to_shared_embedding(eval_pca_matrix)

    print(f"      Projected Text Embeddings Shape: {text_shared_embeds.shape}")
    print(f"      Projected PCA Embeddings Shape:  {pca_shared_embeds.shape}")

    # Save output projected embeddings
    os.makedirs(os.path.dirname(config.paths.embeddings_output), exist_ok=True)
    torch.save(
        {
            "text_shared_embeddings": text_shared_embeds,
            "pca_shared_embeddings": pca_shared_embeds,
            "eval_texts": eval_texts,
            "eval_pca_matrix": eval_pca_matrix,
        },
        config.paths.embeddings_output,
    )
    print(f"      Saved projected shared embeddings to '{config.paths.embeddings_output}'")

    # 5. Predict 5D PCA Components for 1...N Text Entry Records
    print("\n[4/5] Mapping 1...N text entries to predicted 5D PCA Component Space...")
    predicted_pca_matrix = pipeline.predict_pca_components(
        text_list=eval_texts,
        reference_pca_matrix=eval_pca_matrix,
        reference_texts=eval_texts,
    )
    print(f"      Predicted PCA Matrix Shape: {predicted_pca_matrix.shape}")

    # Export predicted PCA scores to CSV
    import pandas as pd
    output_df_dict = {config.dataset.text_column: eval_texts}
    for i, col_name in enumerate(config.dataset.pca_columns):
        output_df_dict[f"Predicted_{col_name}"] = predicted_pca_matrix[:, i]

    pca_csv_output_path = os.path.join(config.paths.output_dir, "predicted_pca_scores.csv")
    pd.DataFrame(output_df_dict).to_csv(pca_csv_output_path, index=False)
    print(f"      Exported predicted 5D PCA scores to CSV: '{pca_csv_output_path}'")

    # Display sample predictions
    print("\n  Sample Text-to-PCA Component Predictions (First 3 entries):")
    for idx in range(min(3, len(eval_texts))):
        scores_fmt = ", ".join([f"{config.dataset.pca_columns[c]}: {predicted_pca_matrix[idx, c]:.3f}" for c in range(len(config.dataset.pca_columns))])
        print(f"    Entry [{idx+1}]: \"{eval_texts[idx][:80]}...\"")
        print(f"            --> [{scores_fmt}]\n")

    # 6. Cross-Modal Retrieval Query Example
    print(f"[5/5] Running Sample Cross-Modal Retrieval Query (Top {args.top_k})...")
    num_pca_dims = config.dataset.pca_input_dim
    sample_query_pca = [1.5 if i == 0 else (-1.0 if i == 1 else 0.5) for i in range(num_pca_dims)]

    print(f"      Target PCA Profile Vector ({num_pca_dims}D): {sample_query_pca}")

    results = pipeline.cross_modal_retrieval(
        query_pca_vector=sample_query_pca,
        text_database=eval_texts,
        top_k=args.top_k,
    )

    print("\n" + "-" * 70)
    print("      Top Retrieved Journal Entries for Target PCA Profile")
    print("-" * 70)
    for res in results:
        print(f"  Rank #{res['rank']} | Sim Score: {res['similarity_score']:.4f} | Entry Index: {res['index']}")
        print(f"  Text: \"{res['text']}\"\n")

    print("=" * 70)
    print(" SUCCESS: Inference, PCA prediction, and retrieval completed cleanly!")
    print("=" * 70)


if __name__ == "__main__":
    main()

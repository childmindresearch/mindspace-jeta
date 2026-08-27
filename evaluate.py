#!/usr/bin/env python3
"""
Test Performance Evaluation Orchestrator.
Evaluates a trained Contrastive Projection Model on a held-out test CSV dataset.
Computes cross-modal retrieval accuracy (Top-1, Top-5, MRR), text-to-PCA prediction accuracy
(MSE, MAE, R^2 variance score per component), and shared space alignment metrics.
Exports test performance report to JSON and CSV.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

import numpy as np
import pandas as pd
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


def compute_mrr(similarity_matrix: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) across similarity matrix diagonal targets."""
    n = similarity_matrix.shape[0]
    ranks = []
    for i in range(n):
        sims = similarity_matrix[i]
        # Rank of target element i
        ranked_indices = np.argsort(-sims)
        rank = np.where(ranked_indices == i)[0][0] + 1  # 1-indexed rank
        ranks.append(1.0 / rank)
    return float(np.mean(ranks))


def compute_topk(similarity_matrix: np.ndarray, k: int = 1) -> float:
    """Computes Top-K retrieval accuracy across similarity matrix diagonal targets."""
    n = similarity_matrix.shape[0]
    k = min(k, n)
    correct = 0
    for i in range(n):
        sims = similarity_matrix[i]
        top_k_indices = np.argsort(-sims)[:k]
        if i in top_k_indices:
            correct += 1
    return float(correct / n)


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes R^2 (coefficient of determination) variance explained score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - (ss_res / ss_tot))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Accuracy on Held-Out Test Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to held-out test CSV dataset")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    print("=" * 70)
    print("      Contrastive Projection Model Test Performance Evaluation")
    print("=" * 70)

    # 1. Load Configuration & Paths
    config = load_config(config_path)
    test_csv_path = args.test_csv or config.dataset.inference_csv_path
    checkpoint_path = config.paths.model_checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Trained model checkpoint not found at '{checkpoint_path}'. Run `python train.py` first.")
        sys.exit(1)

    # Ensure test CSV exists (generate synthetic test data if missing)
    if not os.path.exists(test_csv_path):
        print(f"[Warning] Test CSV '{test_csv_path}' not found. Generating synthetic test dataset (N=100)...")
        generate_synthetic_csv(
            output_csv_path=test_csv_path,
            num_samples=100,
            text_column=config.dataset.text_column,
            pca_columns=config.dataset.pca_columns,
            seed=999,
        )

    # 2. Load Test Dataset
    print(f"[1/4] Reading held-out test dataset from '{test_csv_path}'...")
    test_texts, test_pca_matrix = load_csv_dataset(
        csv_path=test_csv_path,
        text_column=config.dataset.text_column,
        pca_columns=config.dataset.pca_columns,
    )
    n_samples = len(test_texts)
    print(f"      Loaded {n_samples} test records with PCA matrix shape {test_pca_matrix.shape}")

    # 3. Load Model Pipeline
    print(f"[2/4] Loading trained model checkpoint from '{checkpoint_path}'...")
    pipeline = CLIPPCAPipeline.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)

    # 4. Project Test Data to 16D Shared Metric Space
    print("[3/4] Projecting test data & computing evaluation metrics...")
    test_text_shared = pipeline.predict_shared_embedding(test_texts)
    test_pca_shared = pipeline.pca_to_shared_embedding(test_pca_matrix)

    # Cosine Similarity Matrix (N, N)
    sim_matrix_text2pca = np.dot(test_text_shared, test_pca_shared.T)
    sim_matrix_pca2text = sim_matrix_text2pca.T

    # A. Cross-Modal Retrieval Metrics
    top1_txt2pca = compute_topk(sim_matrix_text2pca, k=1)
    top5_txt2pca = compute_topk(sim_matrix_text2pca, k=5)
    mrr_txt2pca = compute_mrr(sim_matrix_text2pca)

    top1_pca2txt = compute_topk(sim_matrix_pca2text, k=1)
    top5_pca2txt = compute_topk(sim_matrix_pca2text, k=5)
    mrr_pca2txt = compute_mrr(sim_matrix_pca2text)

    # B. Text-to-5D PCA Component Prediction Accuracy
    predicted_pca_matrix = pipeline.predict_pca_components(
        text_list=test_texts,
        reference_pca_matrix=test_pca_matrix,
    )

    mse_overall = float(np.mean((test_pca_matrix - predicted_pca_matrix) ** 2))
    rmse_overall = float(np.sqrt(mse_overall))
    mae_overall = float(np.mean(np.abs(test_pca_matrix - predicted_pca_matrix)))
    r2_overall = compute_r2_score(test_pca_matrix, predicted_pca_matrix)

    # Per-component R^2 breakdown
    r2_per_component: Dict[str, float] = {}
    mae_per_component: Dict[str, float] = {}
    for i, col_name in enumerate(config.dataset.pca_columns):
        r2_per_component[col_name] = compute_r2_score(test_pca_matrix[:, i], predicted_pca_matrix[:, i])
        mae_per_component[col_name] = float(np.mean(np.abs(test_pca_matrix[:, i] - predicted_pca_matrix[:, i])))

    # C. Shared Metric Space Alignment Quality
    pos_sims = np.diag(sim_matrix_text2pca)
    neg_mask = ~np.eye(n_samples, dtype=bool)
    neg_sims = sim_matrix_text2pca[neg_mask]

    mean_pos_sim = float(np.mean(pos_sims))
    mean_neg_sim = float(np.mean(neg_sims))
    separation_margin = mean_pos_sim - mean_neg_sim

    # 5. Export Test Evaluation Reports
    logs_dir = config.paths.logs_dir
    os.makedirs(logs_dir, exist_ok=True)

    test_report_json = {
        "test_dataset_path": test_csv_path,
        "n_test_samples": n_samples,
        "model_checkpoint": checkpoint_path,
        "cross_modal_retrieval": {
            "text_to_pca_top1_accuracy": top1_txt2pca,
            "text_to_pca_top5_accuracy": top5_txt2pca,
            "text_to_pca_mrr": mrr_txt2pca,
            "pca_to_text_top1_accuracy": top1_pca2txt,
            "pca_to_text_top5_accuracy": top5_pca2txt,
            "pca_to_text_mrr": mrr_pca2txt,
        },
        "pca_component_prediction": {
            "mse_overall": mse_overall,
            "rmse_overall": rmse_overall,
            "mae_overall": mae_overall,
            "r2_overall": r2_overall,
            "r2_per_component": r2_per_component,
            "mae_per_component": mae_per_component,
        },
        "shared_space_alignment": {
            "mean_positive_cosine_similarity": mean_pos_sim,
            "mean_negative_cosine_similarity": mean_neg_sim,
            "separation_margin": separation_margin,
        },
    }

    json_output_path = os.path.join(logs_dir, "test_evaluation_report.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(test_report_json, f, indent=2)

    # Export CSV summary
    csv_rows = [
        {"Metric Category": "Retrieval (Text->PCA)", "Metric Name": "Top-1 Accuracy", "Value": f"{top1_txt2pca:.2%}"},
        {"Metric Category": "Retrieval (Text->PCA)", "Metric Name": "Top-5 Accuracy", "Value": f"{top5_txt2pca:.2%}"},
        {"Metric Category": "Retrieval (Text->PCA)", "Metric Name": "Mean Reciprocal Rank (MRR)", "Value": f"{mrr_txt2pca:.4f}"},
        {"Metric Category": "Retrieval (PCA->Text)", "Metric Name": "Top-1 Accuracy", "Value": f"{top1_pca2txt:.2%}"},
        {"Metric Category": "Retrieval (PCA->Text)", "Metric Name": "Top-5 Accuracy", "Value": f"{top5_pca2txt:.2%}"},
        {"Metric Category": "Retrieval (PCA->Text)", "Metric Name": "Mean Reciprocal Rank (MRR)", "Value": f"{mrr_pca2txt:.4f}"},
        {"Metric Category": "PCA Prediction", "Metric Name": "Overall R^2 Score", "Value": f"{r2_overall:.4f}"},
        {"Metric Category": "PCA Prediction", "Metric Name": "Overall MAE", "Value": f"{mae_overall:.4f}"},
        {"Metric Category": "PCA Prediction", "Metric Name": "Overall RMSE", "Value": f"{rmse_overall:.4f}"},
        {"Metric Category": "Shared 16D Space", "Metric Name": "Mean Pos Pair Sim", "Value": f"{mean_pos_sim:.4f}"},
        {"Metric Category": "Shared 16D Space", "Metric Name": "Mean Neg Pair Sim", "Value": f"{mean_neg_sim:.4f}"},
        {"Metric Category": "Shared 16D Space", "Metric Name": "Separation Margin", "Value": f"{separation_margin:+.4f}"},
    ]

    csv_output_path = os.path.join(logs_dir, "test_evaluation_report.csv")
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)

    # 6. Display Console Performance Report Card
    print("\n" + "=" * 70)
    print("          TEST PERFORMANCE EVALUATION REPORT CARD")
    print("=" * 70)
    print(f" Test Dataset: '{test_csv_path}' (N={n_samples})")
    print("-" * 70)
    print(" 1. CROSS-MODAL RETRIEVAL ACCURACY:")
    print(f"    - Text -> PCA Top-1 Accuracy:  {top1_txt2pca:.2%}")
    print(f"    - Text -> PCA Top-5 Accuracy:  {top5_txt2pca:.2%}")
    print(f"    - Text -> PCA MRR:             {mrr_txt2pca:.4f}")
    print(f"    - PCA -> Text Top-1 Accuracy:  {top1_pca2txt:.2%}")
    print(f"    - PCA -> Text Top-5 Accuracy:  {top5_pca2txt:.2%}")
    print(f"    - PCA -> Text MRR:             {mrr_pca2txt:.4f}")
    print("-" * 70)
    print(" 2. 5D PCA COMPONENT PREDICTION ACCURACY (Text -> PCA):")
    print(f"    - Overall R^2 Variance Score:  {r2_overall:.4f}")
    print(f"    - Overall MAE:                 {mae_overall:.4f}")
    print(f"    - Overall RMSE:                {rmse_overall:.4f}")
    print("    - Per-Component R^2 Breakdown:")
    for col_name, r2_val in r2_per_component.items():
        print(f"       * {col_name:12s} R^2 = {r2_val:+.4f} | MAE = {mae_per_component[col_name]:.4f}")
    print("-" * 70)
    print(" 3. 16D SHARED METRIC SPACE ALIGNMENT:")
    print(f"    - Mean Matched Pair Sim:       {mean_pos_sim:+.4f}")
    print(f"    - Mean Unmatched Pair Sim:     {mean_neg_sim:+.4f}")
    print(f"    - Separation Margin:           {separation_margin:+.4f}")
    print("=" * 70)
    print(f" SUCCESS: Test evaluation reports saved to:")
    print(f"   - JSON: '{json_output_path}'")
    print(f"   - CSV:  '{csv_output_path}'")
    print("=" * 70)


if __name__ == "__main__":
    main()

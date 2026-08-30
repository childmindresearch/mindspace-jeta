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
from typing import Dict, Any, List, Tuple

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, Config
from src.inference import CLIPPCAPipeline
from src.utils import load_csv_dataset, generate_synthetic_csv


def compute_mrr(similarity_matrix: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) across similarity matrix diagonal targets."""
    n = similarity_matrix.shape[0]
    if n == 0:
        return 0.0
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
    if n == 0:
        return 0.0
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


def compute_profile_shape_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes per-sample Pearson correlation (r) and Cosine Similarity between actual and predicted PCA profiles."""
    n = y_true.shape[0]
    if n == 0:
        return {"mean_sample_pearson_r": 0.0, "mean_sample_cosine_sim": 0.0}

    pearson_rs = []
    cosine_sims = []

    for i in range(n):
        u = y_true[i]
        v = y_pred[i]

        # Cosine similarity
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u > 0 and norm_v > 0:
            cos_sim = np.dot(u, v) / (norm_u * norm_v)
        else:
            cos_sim = 0.0
        cosine_sims.append(cos_sim)

        # Pearson correlation
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        u_dev = u - u_mean
        v_dev = v - v_mean
        denom = np.sqrt(np.sum(u_dev ** 2) * np.sum(v_dev ** 2))
        if denom > 0:
            r = np.sum(u_dev * v_dev) / denom
        else:
            r = 0.0
        pearson_rs.append(r)

    return {
        "mean_sample_pearson_r": float(np.mean(pearson_rs)),
        "mean_sample_cosine_sim": float(np.mean(cosine_sims)),
    }


def compute_dominant_component_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pca_columns: list,
    use_abs: bool = False,
    use_standardized: bool = True,
) -> Dict[str, Any]:
    """Computes Top-1 & Top-2 classification accuracy, confusion matrix, and per-trait F1 metrics for dominant PCA trait identification.
    
    If use_standardized is True, applies column-wise Z-score scaling (z = (y_pred - mu) / sigma) to remove magnitude bias prior to argmax.
    """
    n = y_true.shape[0]
    num_components = len(pca_columns)
    if n == 0:
        return {}

    # Apply column-wise Z-score standardization if requested
    if use_standardized:
        mean_pred = np.mean(y_pred, axis=0, keepdims=True)
        std_pred = np.std(y_pred, axis=0, keepdims=True)
        std_pred = np.where(std_pred == 0, 1e-8, std_pred)
        y_pred_eval = (y_pred - mean_pred) / std_pred
    else:
        y_pred_eval = y_pred

    # Extract dominant component index for ground truth and predictions
    if use_abs:
        y_true_dom = np.argmax(np.abs(y_true), axis=1)
        y_pred_dom = np.argmax(np.abs(y_pred_eval), axis=1)
    else:
        y_true_dom = np.argmax(y_true, axis=1)
        y_pred_dom = np.argmax(y_pred_eval, axis=1)

    # Top-1 Dominant Accuracy
    top1_correct = np.sum(y_true_dom == y_pred_dom)
    top1_acc = float(top1_correct / n)

    # Top-2 Dominant Accuracy
    top2_correct = 0
    for i in range(n):
        if use_abs:
            top2_indices = np.argsort(-np.abs(y_pred_eval[i]))[:2]
        else:
            top2_indices = np.argsort(-y_pred_eval[i])[:2]
        if y_true_dom[i] in top2_indices:
            top2_correct += 1
    top2_acc = float(top2_correct / n)

    # 5x5 Confusion Matrix (rows: true, cols: predicted)
    cm = np.zeros((num_components, num_components), dtype=int)
    for t, p in zip(y_true_dom, y_pred_dom):
        cm[t, p] += 1

    # Per-component Precision, Recall, F1
    per_component_report: Dict[str, Dict[str, float]] = {}
    for c, col_name in enumerate(pca_columns):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = int(np.sum(cm[c, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_component_report[col_name] = {
            "support": support,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    random_baseline_top1 = 1.0 / num_components
    random_baseline_top2 = min(1.0, 2.0 / num_components)

    criterion_name = ("standardized_" if use_standardized else "raw_") + ("absolute_value" if use_abs else "signed_value")

    return {
        "dominant_criterion": criterion_name,
        "use_standardized_argmax": use_standardized,
        "top1_dominant_accuracy": top1_acc,
        "top2_dominant_accuracy": top2_acc,
        "random_baseline_top1": random_baseline_top1,
        "random_baseline_top2": random_baseline_top2,
        "confusion_matrix": cm.tolist(),
        "per_component_report": per_component_report,
    }


def plot_evaluation_diagnostics(
    test_pca_matrix: np.ndarray,
    predicted_pca_matrix: np.ndarray,
    test_text_shared: np.ndarray,
    pca_columns: List[str],
    dominant_metrics: Dict[str, Any],
    r2_per_component: Dict[str, float],
    mae_per_component: Dict[str, float],
    r2_overall: float,
    visuals_dir: str = "./outputs/visuals",
) -> None:
    """Generates 4 high-resolution diagnostic plots supporting researcher interpretation."""
    os.makedirs(visuals_dir, exist_ok=True)
    num_components = len(pca_columns)

    # 1. Confusion Matrix Heatmap
    cm = np.array(dominant_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9, 7))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    total_samples = np.sum(cm)
    for i in range(num_components):
        for j in range(num_components):
            count = cm[i, j]
            pct = (count / total_samples * 100) if total_samples > 0 else 0.0
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color="white" if count > (np.max(cm) / 2) else "black", fontsize=9)

    ax.set_xticks(range(num_components))
    ax.set_yticks(range(num_components))
    ax.set_xticklabels(pca_columns, rotation=35, ha="left", fontsize=9)
    ax.set_yticklabels(pca_columns, fontsize=9)
    ax.set_xlabel("Predicted Dominant Component", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Dominant Component", fontweight="bold")
    ax.set_title(f"Dominant Trait Confusion Matrix ({dominant_metrics['dominant_criterion'].upper()})", fontweight="bold", pad=20)
    plt.tight_layout()
    cm_path = os.path.join(visuals_dir, "confusion_matrix_heatmap.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. Scatter Plot Grid: Actual vs Predicted
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for i, col_name in enumerate(pca_columns):
        ax = axes_flat[i]
        actual = test_pca_matrix[:, i]
        pred = predicted_pca_matrix[:, i]

        ax.scatter(actual, pred, alpha=0.6, color="#1f77b4", edgecolors="none", s=30)
        
        min_val = min(np.min(actual), np.min(pred)) - 0.5
        max_val = max(np.max(actual), np.max(pred)) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="Ideal (y = x)")

        r2_val = r2_per_component.get(col_name, 0.0)
        mae_val = mae_per_component.get(col_name, 0.0)
        ax.set_title(f"{col_name}\n(R² = {r2_val:+.3f} | MAE = {mae_val:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.grid(True, alpha=0.3)

    # Panel 6: Overall Combined Scatter
    ax_last = axes_flat[5]
    ax_last.scatter(test_pca_matrix.flatten(), predicted_pca_matrix.flatten(), alpha=0.4, color="#2ca02c", s=20)
    all_min = min(np.min(test_pca_matrix), np.min(predicted_pca_matrix)) - 0.5
    all_max = max(np.max(test_pca_matrix), np.max(predicted_pca_matrix)) + 0.5
    ax_last.plot([all_min, all_max], [all_min, all_max], "r--", alpha=0.7, label="Ideal (y = x)")
    ax_last.set_title(f"Overall All Components\n(Overall R² = {r2_overall:+.3f})", fontsize=10, fontweight="bold")
    ax_last.set_xlabel("Actual Score (All)")
    ax_last.set_ylabel("Predicted Score (All)")
    ax_last.grid(True, alpha=0.3)

    plt.suptitle("PCA Component Score Prediction: Actual vs. Predicted", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    scatter_path = os.path.join(visuals_dir, "pca_actual_vs_predicted.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    # 3. Radar Chart: Representative Sample Profiles
    num_vars = num_components
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    sample_indices = [0, min(1, len(test_pca_matrix) - 1)]
    colors_actual = ["#1f77b4", "#ff7f0e"]
    colors_pred = ["#aec7e8", "#ffbb78"]

    for idx, sample_i in enumerate(sample_indices):
        act_vals = test_pca_matrix[sample_i].tolist()
        pred_vals = predicted_pca_matrix[sample_i].tolist()
        act_vals += act_vals[:1]
        pred_vals += pred_vals[:1]

        ax.plot(angles, act_vals, linewidth=2, linestyle="solid", color=colors_actual[idx], label=f"Sample #{sample_i+1} Actual")
        ax.plot(angles, pred_vals, linewidth=2, linestyle="dashed", color=colors_pred[idx], label=f"Sample #{sample_i+1} Predicted")
        ax.fill(angles, act_vals, color=colors_actual[idx], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pca_columns, fontsize=9)
    ax.set_title("Representative 5D Profile Shape Fidelity (Actual vs. Predicted)", fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    plt.tight_layout()
    radar_path = os.path.join(visuals_dir, "sample_profile_radar.png")
    plt.savefig(radar_path, dpi=200)
    plt.close()

    # 4. Shared Metric Space 2D Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if test_text_shared.shape[0] >= 2:
        u, s, vh = np.linalg.svd(test_text_shared - np.mean(test_text_shared, axis=0), full_matrices=False)
        coords_2d = u[:, :2] * s[:2]
    else:
        coords_2d = test_text_shared[:, :2]

    use_abs = (dominant_metrics["dominant_criterion"] == "absolute_value")
    dom_labels = np.argmax(np.abs(test_pca_matrix), axis=1) if use_abs else np.argmax(test_pca_matrix, axis=1)

    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=dom_labels, cmap="Set1", alpha=0.8, s=40)
    cbar = plt.colorbar(scatter, ticks=range(num_components))
    cbar.ax.set_yticklabels(pca_columns)
    ax.set_title("16D Shared Space 2D Projection (Colored by True Dominant Trait)", fontweight="bold")
    ax.set_xlabel("Shared Vector Comp 1")
    ax.set_ylabel("Shared Vector Comp 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    clusters_path = os.path.join(visuals_dir, "shared_space_clusters.png")
    plt.savefig(clusters_path, dpi=200)
    plt.close()

    print(f"\n[Visuals Engine] Generated 4 diagnostic evaluation plots in '{visuals_dir}':")
    print(f"   - Confusion Heatmap:      '{cm_path}'")
    print(f"   - Actual vs Predicted:    '{scatter_path}'")
    print(f"   - Profile Radar Chart:    '{radar_path}'")
    print(f"   - Shared Space 2D Map:    '{clusters_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Accuracy on Held-Out Test Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to held-out test CSV dataset")
    parser.add_argument("--use_abs_dominant", action="store_true", help="Use absolute value magnitude to define dominant PCA component")
    parser.add_argument("--use_standardized_dominant", dest="use_standardized_dominant", action="store_true", default=None, help="Force Z-score standardization before dominant PCA argmax")
    parser.add_argument("--no_standardized_dominant", dest="use_standardized_dominant", action="store_false", default=None, help="Disable Z-score standardization before dominant PCA argmax")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate synthetic CSV dataset if test file is missing")
    args = parser.parse_args()

    config_path = args.config

    print("=" * 70)
    print("      Contrastive Projection Model Test Performance Evaluation")
    print("=" * 70)

    # 1. Load Configuration & Paths
    config = load_config(config_path)
    test_csv_path = args.test_csv or config.dataset.inference_csv_path
    checkpoint_path = config.paths.model_checkpoint

    # Resolve Z-score standardization toggle from CLI override or config
    if args.use_standardized_dominant is not None:
        use_standardized_dominant = args.use_standardized_dominant
    else:
        use_standardized_dominant = config.evaluation.use_standardized_dominant_argmax

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Trained model checkpoint not found at '{checkpoint_path}'. Run `python train.py` first.")
        sys.exit(1)

    # Ensure test CSV exists (generate synthetic test data if requested)
    if not os.path.exists(test_csv_path):
        if args.generate_synthetic:
            print(f"[Synthetic Data] Test CSV '{test_csv_path}' not found. Generating synthetic test dataset (N=100)...")
            generate_synthetic_csv(
                output_csv_path=test_csv_path,
                text_column=config.dataset.text_column,
                pca_columns=config.dataset.pca_columns,
                num_samples=100,
                seed=999,
            )
        else:
            print(f"[Dataset Error] Test dataset file not found at '{test_csv_path}'.")
            print("Please check 'dataset.inference_csv_path' in 'config.yaml' or pass '--generate_synthetic' to create a test dataset.")
            sys.exit(1)

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

    # C. Profile Shape Correlation & Fidelity
    shape_metrics = compute_profile_shape_correlation(test_pca_matrix, predicted_pca_matrix)

    # D. Dominant PCA Component Classification
    dominant_metrics = compute_dominant_component_metrics(
        y_true=test_pca_matrix,
        y_pred=predicted_pca_matrix,
        pca_columns=config.dataset.pca_columns,
        use_abs=args.use_abs_dominant,
        use_standardized=use_standardized_dominant,
    )

    # E. Shared Metric Space Alignment Quality
    pos_sims = np.diag(sim_matrix_text2pca)
    neg_mask = ~np.eye(n_samples, dtype=bool)
    neg_sims = sim_matrix_text2pca[neg_mask]

    mean_pos_sim = float(np.mean(pos_sims))
    mean_neg_sim = float(np.mean(neg_sims))
    separation_margin = mean_pos_sim - mean_neg_sim

    # 5. Export Test Evaluation Reports
    logs_dir = config.paths.logs_dir
    os.makedirs(logs_dir, exist_ok=True)

    # A. Build Side-by-Side Predictions vs. Actuals DataFrame
    pred_vs_actual_dict = {config.dataset.text_column: test_texts}
    for i, col_name in enumerate(config.dataset.pca_columns):
        actual_vals = test_pca_matrix[:, i]
        pred_vals = predicted_pca_matrix[:, i]
        abs_errors = np.abs(actual_vals - pred_vals)
        pred_vs_actual_dict[f"Actual_{col_name}"] = actual_vals
        pred_vs_actual_dict[f"Predicted_{col_name}"] = pred_vals
        pred_vs_actual_dict[f"Abs_Error_{col_name}"] = abs_errors

    pred_vs_actual_dict["Metric_Space_Similarity"] = pos_sims
    pred_vs_actual_df = pd.DataFrame(pred_vs_actual_dict)

    pred_vs_actual_csv_path = os.path.join(logs_dir, "test_predictions_vs_actuals.csv")
    pred_vs_actual_df.to_csv(pred_vs_actual_csv_path, index=False)

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
        "profile_shape_fidelity": shape_metrics,
        "dominant_component_classification": dominant_metrics,
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
        {"Metric Category": "Profile Shape", "Metric Name": "Mean Sample Pearson r", "Value": f"{shape_metrics['mean_sample_pearson_r']:.4f}"},
        {"Metric Category": "Profile Shape", "Metric Name": "Mean Sample Cosine Sim", "Value": f"{shape_metrics['mean_sample_cosine_sim']:.4f}"},
        {"Metric Category": "Dominant Trait Class", "Metric Name": "Top-1 Dominant Accuracy", "Value": f"{dominant_metrics['top1_dominant_accuracy']:.2%} (vs {dominant_metrics['random_baseline_top1']:.2%} rand)"},
        {"Metric Category": "Dominant Trait Class", "Metric Name": "Top-2 Dominant Accuracy", "Value": f"{dominant_metrics['top2_dominant_accuracy']:.2%} (vs {dominant_metrics['random_baseline_top2']:.2%} rand)"},
        {"Metric Category": f"Shared {config.model_architecture.shared_dim}D Space", "Metric Name": "Mean Pos Pair Sim", "Value": f"{mean_pos_sim:.4f}"},
        {"Metric Category": f"Shared {config.model_architecture.shared_dim}D Space", "Metric Name": "Mean Neg Pair Sim", "Value": f"{mean_neg_sim:.4f}"},
        {"Metric Category": f"Shared {config.model_architecture.shared_dim}D Space", "Metric Name": "Separation Margin", "Value": f"{separation_margin:+.4f}"},
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
    print(" 3. PROFILE SHAPE FIDELITY (Per-Sample Correlation):")
    print(f"    - Mean Per-Sample Pearson r:   {shape_metrics['mean_sample_pearson_r']:+.4f}")
    print(f"    - Mean Per-Sample Cosine Sim:  {shape_metrics['mean_sample_cosine_sim']:+.4f}")
    print("-" * 70)
    print(f" 4. DOMINANT PCA COMPONENT CLASSIFICATION ({dominant_metrics['dominant_criterion'].upper()}):")
    print(f"    - Top-1 Dominant Trait Acc:    {dominant_metrics['top1_dominant_accuracy']:.2%}  (Random Baseline: {dominant_metrics['random_baseline_top1']:.2%})")
    print(f"    - Top-2 Dominant Trait Acc:    {dominant_metrics['top2_dominant_accuracy']:.2%}  (Random Baseline: {dominant_metrics['random_baseline_top2']:.2%})")
    print("    - Per-Trait Precision / Recall / F1 Breakdown:")
    for col_name, p_metrics in dominant_metrics["per_component_report"].items():
        print(
            f"       * {col_name:12s} | N={p_metrics['support']:2d} | "
            f"Prec={p_metrics['precision']:.2f} | Rec={p_metrics['recall']:.2f} | F1={p_metrics['f1_score']:.2f}"
        )
    print("\n    - 5x5 Confusion Matrix (Rows: True Dominant, Cols: Predicted Dominant):")
    cm_df = pd.DataFrame(dominant_metrics["confusion_matrix"], index=config.dataset.pca_columns, columns=config.dataset.pca_columns)
    for line in cm_df.to_string().split("\n"):
        print(f"       {line}")
    print("-" * 70)
    print(f" 5. {config.model_architecture.shared_dim}D SHARED METRIC SPACE ALIGNMENT:")
    print(f"    - Mean Matched Pair Sim:       {mean_pos_sim:+.4f}")
    print(f"    - Mean Unmatched Pair Sim:     {mean_neg_sim:+.4f}")
    print(f"    - Separation Margin:           {separation_margin:+.4f}")
    print("-" * 70)
    print(" 6. SAMPLE PREDICTIONS VS. ACTUALS (First 3 Test Entries):")
    for idx in range(min(3, n_samples)):
        print(f"    Entry [{idx+1}]: \"{test_texts[idx][:75]}...\"")
        for c, col_name in enumerate(config.dataset.pca_columns):
            act = test_pca_matrix[idx, c]
            prd = predicted_pca_matrix[idx, c]
            err = abs(act - prd)
            print(f"       * {col_name:8s} | Actual: {act:+.3f} | Predicted: {prd:+.3f} | Abs Err: {err:.3f}")
        print()
    visuals_dir = os.path.join(config.paths.output_dir, "visuals")
    plot_evaluation_diagnostics(
        test_pca_matrix=test_pca_matrix,
        predicted_pca_matrix=predicted_pca_matrix,
        test_text_shared=test_text_shared,
        pca_columns=config.dataset.pca_columns,
        dominant_metrics=dominant_metrics,
        r2_per_component=r2_per_component,
        mae_per_component=mae_per_component,
        r2_overall=r2_overall,
        visuals_dir=visuals_dir,
    )

    print("=" * 70)
    print(f" SUCCESS: Test evaluation reports & figures saved to:")
    print(f"   - Predictions vs Actuals CSV: '{pred_vs_actual_csv_path}'")
    print(f"   - Metrics Summary JSON:       '{json_output_path}'")
    print(f"   - Metrics Summary CSV:        '{csv_output_path}'")
    print(f"   - Diagnostic Plots Directory: '{visuals_dir}'")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Training and Validation module implementing Symmetric CLIP / InfoNCE Cross-Entropy loss,
PyTorch DataLoader optimization pipeline using AdamW, detailed epoch metrics logging,
and diagnostic visualization plots to detect overfitting and distribution behavior.
"""

import json
import os
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt

from src.config import Config
from src.dataset import JournalPCADataset
from src.models import ContrastiveProjectionModel


def compute_topk_accuracy(logits: torch.Tensor, k: int = 1) -> float:
    """Computes Top-K contrastive accuracy over mini-batch logits."""
    batch_size = logits.size(0)
    targets = torch.arange(batch_size, device=logits.device)
    k = min(k, batch_size)
    _, top_k_indices = logits.topk(k, dim=-1)
    correct = top_k_indices.eq(targets.view(-1, 1)).sum().item()
    return float(correct / batch_size)


class SymmetricCLIPLoss(nn.Module):
    """Symmetric CLIP / InfoNCE Cross-Entropy Loss over cosine similarity matrices."""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self, text_shared: torch.Tensor, pca_shared: torch.Tensor, logit_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Computes symmetric InfoNCE loss and top-K accuracy metrics across mini-batch."""
        device = text_shared.device
        batch_size = text_shared.size(0)

        # Pairwise cosine similarities scaled by logit scale
        logits_per_text = logit_scale * (text_shared @ pca_shared.T)  # (B, B)
        logits_per_pca = logits_per_text.T                             # (B, B)

        # Ground truth diagonal targets
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        loss_text = self.cross_entropy(logits_per_text, labels)
        loss_pca = self.cross_entropy(logits_per_pca, labels)

        total_loss = (loss_text + loss_pca) / 2.0

        # Compute accuracy metrics
        top1_text = compute_topk_accuracy(logits_per_text, k=1)
        top5_text = compute_topk_accuracy(logits_per_text, k=min(5, batch_size))
        top1_pca = compute_topk_accuracy(logits_per_pca, k=1)
        top5_pca = compute_topk_accuracy(logits_per_pca, k=min(5, batch_size))

        metrics = {
            "top1_acc_text": top1_text,
            "top5_acc_text": top5_text,
            "top1_acc_pca": top1_pca,
            "top5_acc_pca": top5_pca,
        }

        return total_loss, loss_text, loss_pca, metrics


class Trainer:
    """Trainer pipeline managing data loaders, AdamW optimization, training loop, detailed epoch logging, and diagnostic plotting."""

    def __init__(self, model: ContrastiveProjectionModel, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = SymmetricCLIPLoss()

        # Optimizer: AdamW with lr=1e-3, weight_decay=1e-2
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        print(f"[Trainer] Initialized on device: '{self.device}'. Optimizer: AdamW(lr={self.config.training.learning_rate}, weight_decay={self.config.training.weight_decay})")

    def train(self, dataset: JournalPCADataset) -> Dict[str, List[Any]]:
        """Executes 80/20 train/val split, runs training epochs, records metrics CSV & plots diagnostic loss curves."""
        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)

        # Create logs output directory
        logs_dir = self.config.paths.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.paths.model_checkpoint), exist_ok=True)

        # 80/20 Train / Validation split
        total_len = len(dataset)
        train_size = int(total_len * self.config.training.train_split)
        val_size = total_len - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = min(self.config.training.batch_size, train_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=(train_size > batch_size))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"[Trainer] Dataset split: Train={train_size} samples, Val={val_size} samples (Batch Size={batch_size})")

        history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_text_loss": [],
            "train_pca_loss": [],
            "val_text_loss": [],
            "val_pca_loss": [],
            "val_top1_acc_text": [],
            "val_top5_acc_text": [],
            "val_top1_acc_pca": [],
            "val_top5_acc_pca": [],
            "overfitting_gap": [],
            "temperature": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0

        # For cosine similarity distribution analysis
        val_pos_sims: List[float] = []
        val_neg_sims: List[float] = []

        epochs = self.config.training.epochs
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()
            train_losses, train_text_losses, train_pca_losses = [], [], []

            for text_batch, pca_batch in train_loader:
                text_batch = text_batch.to(self.device)
                pca_batch = pca_batch.to(self.device)

                self.optimizer.zero_grad()
                text_shared, pca_shared, logit_scale = self.model(text_batch, pca_batch)

                loss, loss_t, loss_p, _ = self.criterion(text_shared, pca_shared, logit_scale)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_text_losses.append(loss_t.item())
                train_pca_losses.append(loss_p.item())

            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            avg_train_text = float(np.mean(train_text_losses)) if train_text_losses else 0.0
            avg_train_pca = float(np.mean(train_pca_losses)) if train_pca_losses else 0.0

            # Validation phase
            self.model.eval()
            val_losses, val_text_losses, val_pca_losses = [], [], []
            val_t1_txt, val_t5_txt, val_t1_pca, val_t5_pca = [], [], [], []

            with torch.no_grad():
                for text_batch, pca_batch in val_loader:
                    text_batch = text_batch.to(self.device)
                    pca_batch = pca_batch.to(self.device)

                    text_shared, pca_shared, logit_scale = self.model(text_batch, pca_batch)
                    val_loss, loss_t, loss_p, acc_metrics = self.criterion(text_shared, pca_shared, logit_scale)

                    val_losses.append(val_loss.item())
                    val_text_losses.append(loss_t.item())
                    val_pca_losses.append(loss_p.item())
                    val_t1_txt.append(acc_metrics["top1_acc_text"])
                    val_t5_txt.append(acc_metrics["top5_acc_text"])
                    val_t1_pca.append(acc_metrics["top1_acc_pca"])
                    val_t5_pca.append(acc_metrics["top5_acc_pca"])

                    # Store similarity distributions on last epoch
                    if epoch == epochs:
                        sims = (text_shared @ pca_shared.T).cpu().numpy()
                        b_sz = sims.shape[0]
                        pos_s = np.diag(sims)
                        neg_s = sims[~np.eye(b_sz, dtype=bool)]
                        val_pos_sims.extend(pos_s.tolist())
                        val_neg_sims.extend(neg_s.tolist())

            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            avg_val_text = float(np.mean(val_text_losses)) if val_text_losses else 0.0
            avg_val_pca = float(np.mean(val_pca_losses)) if val_pca_losses else 0.0
            avg_t1_txt = float(np.mean(val_t1_txt)) if val_t1_txt else 0.0
            avg_t5_txt = float(np.mean(val_t5_txt)) if val_t5_txt else 0.0
            avg_t1_pca = float(np.mean(val_t1_pca)) if val_t1_pca else 0.0
            avg_t5_pca = float(np.mean(val_t5_pca)) if val_t5_pca else 0.0

            curr_temp = (1.0 / self.model.logit_scale.exp()).item()
            overfitting_gap = avg_val_loss - avg_train_loss

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_text_loss"].append(avg_train_text)
            history["train_pca_loss"].append(avg_train_pca)
            history["val_text_loss"].append(avg_val_text)
            history["val_pca_loss"].append(avg_val_pca)
            history["val_top1_acc_text"].append(avg_t1_txt)
            history["val_top5_acc_text"].append(avg_t5_txt)
            history["val_top1_acc_pca"].append(avg_t1_pca)
            history["val_top5_acc_pca"].append(avg_t5_pca)
            history["overfitting_gap"].append(overfitting_gap)
            history["temperature"].append(curr_temp)

            # Checkpointing best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.config,
                        "epoch": epoch,
                        "val_loss": avg_val_loss,
                    },
                    self.config.paths.model_checkpoint,
                )

            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Top1 Acc: {avg_t1_txt:.2%} | Gap: {overfitting_gap:+.4f}"
                )

        # 1. Export Detailed Training Metrics CSV
        metrics_csv_path = os.path.join(logs_dir, "training_metrics.csv")
        pd.DataFrame(history).to_csv(metrics_csv_path, index=False)
        print(f"[Trainer] Exported detailed training metrics to '{metrics_csv_path}'")

        # 2. Export Training Summary JSON
        summary_json_path = os.path.join(logs_dir, "training_summary.json")
        summary_data = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_overfitting_gap": history["overfitting_gap"][-1],
            "final_val_top1_accuracy": history["val_top1_acc_text"][-1],
            "final_val_top5_accuracy": history["val_top5_acc_text"][-1],
            "final_temperature": history["temperature"][-1],
            "hyperparameters": {
                "learning_rate": self.config.training.learning_rate,
                "weight_decay": self.config.training.weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "shared_dim": self.config.model_architecture.shared_dim,
                "text_head_dims": self.config.model_architecture.text_head_hidden_dims,
                "pca_head_dims": self.config.model_architecture.pca_head_hidden_dims,
            },
        }
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[Trainer] Saved training summary JSON to '{summary_json_path}'")

        # 3. Generate Diagnostic Plots (Loss curves, Overfitting Gap, Accuracy, Similarity Distribution)
        self._plot_diagnostics(history, val_pos_sims, val_neg_sims, logs_dir)

        print(f"[Trainer] Training complete. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}. Model saved to '{self.config.paths.model_checkpoint}'")
        return history

    def _plot_diagnostics(
        self,
        history: Dict[str, List[Any]],
        val_pos_sims: List[float],
        val_neg_sims: List[float],
        logs_dir: str,
    ) -> None:
        """Generates 4-panel diagnostic visualization chart saved into logs directory."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Contrastive Projection Model Training Diagnostics", fontsize=16, fontweight="bold")

        epochs = history["epoch"]

        # Panel 1: Loss & Overfitting Gap
        ax1 = axes[0, 0]
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        ax1.plot(epochs, history["overfitting_gap"], label="Overfitting Gap (Val - Train)", color="#d62728", linewidth=1.5, linestyle=":")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("InfoNCE Loss & Overfitting Gap", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Validation Accuracy (Top-1 and Top-5)
        ax2 = axes[0, 1]
        ax2.plot(epochs, history["val_top1_acc_text"], label="Text->PCA Top-1 Acc", color="#2ca02c", linewidth=2)
        ax2.plot(epochs, history["val_top5_acc_text"], label="Text->PCA Top-5 Acc", color="#98df8a", linewidth=2, linestyle="--")
        ax2.plot(epochs, history["val_top1_acc_pca"], label="PCA->Text Top-1 Acc", color="#9467bd", linewidth=2)
        ax2.plot(epochs, history["val_top5_acc_pca"], label="PCA->Text Top-5 Acc", color="#c5b0d5", linewidth=2, linestyle="--")
        ax2.set_title("Validation Retrieval Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Temperature Scale Progression
        ax3 = axes[1, 0]
        ax3.plot(epochs, history["temperature"], label="Temperature (1 / exp(logit_scale))", color="#e377c2", linewidth=2)
        ax3.set_title("Learnable Temperature Progression", fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Temperature Scale")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Shared Space Cosine Similarity Distribution
        ax4 = axes[1, 1]
        if val_pos_sims and val_neg_sims:
            ax4.hist(val_pos_sims, bins=25, alpha=0.7, label="Positive Pairs (Matched)", color="#2ca02c", density=True)
            ax4.hist(val_neg_sims, bins=25, alpha=0.5, label="Negative Pairs (Unmatched)", color="#d62728", density=True)
            ax4.set_title("Shared 16D Space Cosine Sim Distribution", fontweight="bold")
            ax4.set_xlabel("Cosine Similarity")
            ax4.set_ylabel("Density")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No distribution data", ha="center", va="center")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_output_path = os.path.join(logs_dir, "loss_curves.png")
        plt.savefig(plot_output_path, dpi=200)
        plt.close()
        print(f"[Trainer] Diagnostic loss curves plot saved to '{plot_output_path}'")

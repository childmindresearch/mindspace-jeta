"""
PyTorch Neural Network models for Contrastive Projection alignment.
Implements dynamic Text and PCA Projection Heads into a shared unit hypersphere metric space.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextProjectionHead(nn.Module):
    """Dynamic neural network projecting text embeddings (e.g., 384D or 1024D) to shared space (16D).

    Default architecture: Linear(384 -> 128) -> BatchNorm1d(128) -> ReLU() -> Dropout(p=0.3) -> Linear(128 -> 16)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]

        layers: List[nn.Module] = []
        curr_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            curr_dim = hidden_dim

        # Final projection layer to shared_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PCAProjectionHead(nn.Module):
    """Dynamic neural network projecting PCA score vectors (e.g., 5D) to shared space (16D).

    Default architecture: Linear(5 -> 32) -> BatchNorm1d(32) -> ReLU() -> Linear(32 -> 16)
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 16,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32]

        layers: List[nn.Module] = []
        curr_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim

        # Final projection layer to shared_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContrastiveProjectionModel(nn.Module):
    """CLIP-style Contrastive Projection Model aligning free-text embeddings with PCA score vectors.

    Projects both modalities into a shared L2-normalized metric space (default 16D) with a learnable logit temperature scale.
    """

    def __init__(
        self,
        text_input_dim: int = 384,
        pca_input_dim: int = 5,
        shared_dim: int = 16,
        text_head_hidden_dims: Optional[List[int]] = None,
        text_head_dropout: float = 0.3,
        pca_head_hidden_dims: Optional[List[int]] = None,
        initial_temperature: float = 0.07,
    ):
        super().__init__()
        self.text_input_dim = text_input_dim
        self.pca_input_dim = pca_input_dim
        self.shared_dim = shared_dim

        # Text and PCA projection heads
        self.text_head = TextProjectionHead(
            input_dim=text_input_dim,
            hidden_dims=text_head_hidden_dims or [128],
            output_dim=shared_dim,
            dropout_p=text_head_dropout,
        )

        self.pca_head = PCAProjectionHead(
            input_dim=pca_input_dim,
            hidden_dims=pca_head_hidden_dims or [32],
            output_dim=shared_dim,
        )

        # Learnable logit scale parameter initialized to log(1 / initial_temperature)
        init_scale = math.log(1.0 / initial_temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Projects raw text embeddings to 16D shared space and applies L2 normalization."""
        projected = self.text_head(text_embeds)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def encode_pca(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Projects PCA score matrix to 16D shared space and applies L2 normalization."""
        projected = self.pca_head(pca_scores)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def forward(self, text_embeds: torch.Tensor, pca_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for mini-batch training.

        Returns:
            Tuple of (text_shared_embeds, pca_shared_embeds, logit_scale)
        """
        text_shared = self.encode_text(text_embeds)
        pca_shared = self.encode_pca(pca_scores)

        # Clamp logit scale to prevent numerical instability (max exp scale = 100.0)
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100.0))

        logit_scale_exp = self.logit_scale.exp()
        return text_shared, pca_shared, logit_scale_exp

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


class FourierFeatureExpansion(nn.Module):
    """Random Fourier Feature (RFF) Positional Encoding expanding 5D PCA vectors into 128D periodic features."""

    def __init__(self, input_dim: int, num_features: int = 128, sigma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        # Random projection matrix B drawn from Gaussian distribution (non-trainable)
        B = torch.randn(input_dim, num_features // 2) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SwiGLUResidualBlock(nn.Module):
    """SwiGLU Gated Residual Block with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.w_gate = nn.Linear(in_dim, out_dim)
        self.w_value = nn.Linear(in_dim, out_dim)
        self.w_out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()

        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        gated = F.silu(self.w_gate(normed)) * self.w_value(normed)
        out = self.w_out(self.dropout(gated))
        return out + self.shortcut(x)


class TextProjectionHead(nn.Module):
    """Dynamic neural network projecting text embeddings to shared metric space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 64,
        dropout_p: float = 0.2,
        use_swiglu_residual: bool = True,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("[Model Error] 'text_head_hidden_dims' must be explicitly provided as a non-empty list of layer sizes.")

        layers: List[nn.Module] = []
        curr_dim = input_dim

        for hidden_dim in hidden_dims:
            if use_swiglu_residual:
                layers.append(SwiGLUResidualBlock(curr_dim, hidden_dim, dropout_p=dropout_p))
            else:
                layers.append(nn.Linear(curr_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
            curr_dim = hidden_dim

        # Final projection layer to shared_dim
        layers.append(nn.LayerNorm(curr_dim))
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PCAProjectionHead(nn.Module):
    """Dynamic neural network projecting PCA score vectors to shared metric space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 64,
        use_rff: bool = True,
        rff_dim: int = 128,
        use_swiglu_residual: bool = True,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("[Model Error] 'pca_head_hidden_dims' must be explicitly provided as a non-empty list of layer sizes.")

        self.use_rff = use_rff
        if use_rff:
            self.rff = FourierFeatureExpansion(input_dim=input_dim, num_features=rff_dim)
            curr_dim = rff_dim
        else:
            self.rff = None
            curr_dim = input_dim

        layers: List[nn.Module] = []

        for hidden_dim in hidden_dims:
            if use_swiglu_residual:
                layers.append(SwiGLUResidualBlock(curr_dim, hidden_dim, dropout_p=0.0))
            else:
                layers.append(nn.Linear(curr_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
            curr_dim = hidden_dim

        # Final projection layer to shared_dim
        layers.append(nn.LayerNorm(curr_dim))
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rff and self.rff is not None:
            x = self.rff(x)
        return self.network(x)


class ContrastiveProjectionModel(nn.Module):
    """Post-CLIP Contrastive Projection Model supporting SigLIP loss, RFF expansion, SwiGLU residual heads, and multi-task PCA prediction.

    Projects both modalities into a shared L2-normalized metric space with learnable logit scale and bias.
    """

    def __init__(
        self,
        text_input_dim: int,
        pca_input_dim: int,
        text_head_hidden_dims: List[int],
        pca_head_hidden_dims: List[int],
        shared_dim: int = 64,
        text_head_dropout: float = 0.2,
        initial_temperature: float = 0.1,
        use_rff_expansion: bool = True,
        rff_dim: int = 128,
        use_swiglu_residual: bool = True,
    ):
        super().__init__()
        self.text_input_dim = text_input_dim
        self.pca_input_dim = pca_input_dim
        self.shared_dim = shared_dim

        # Text and PCA projection heads
        self.text_head = TextProjectionHead(
            input_dim=text_input_dim,
            hidden_dims=text_head_hidden_dims,
            output_dim=shared_dim,
            dropout_p=text_head_dropout,
            use_swiglu_residual=use_swiglu_residual,
        )

        self.pca_head = PCAProjectionHead(
            input_dim=pca_input_dim,
            hidden_dims=pca_head_hidden_dims,
            output_dim=shared_dim,
            use_rff=use_rff_expansion,
            rff_dim=rff_dim,
            use_swiglu_residual=use_swiglu_residual,
        )

        # Auxiliary direct 5D PCA decoder from text shared embedding
        self.aux_pca_decoder = nn.Linear(shared_dim, pca_input_dim)

        # SigLIP learnable logit scale and bias parameters
        init_scale = math.log(1.0 / initial_temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * (-10.0))

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Projects raw text embeddings to shared metric space and applies L2 normalization."""
        projected = self.text_head(text_embeds)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def encode_pca(self, pca_scores: torch.Tensor) -> torch.Tensor:
        """Projects PCA score matrix to shared metric space and applies L2 normalization."""
        projected = self.pca_head(pca_scores)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def predict_pca_from_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Directly decodes predicted 5D PCA scores from text embeddings using text_head + aux_pca_decoder."""
        text_shared = self.encode_text(text_embeds)
        return self.aux_pca_decoder(text_shared)

    def forward(
        self, text_embeds: torch.Tensor, pca_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for mini-batch training.

        Returns:
            Tuple of (text_shared, pca_shared, logit_scale, logit_bias, aux_pca_pred)
        """
        text_shared = self.encode_text(text_embeds)
        pca_shared = self.encode_pca(pca_scores)
        aux_pca_pred = self.aux_pca_decoder(text_shared)

        # Clamp logit scale for numerical stability
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100.0))

        logit_scale_exp = self.logit_scale.exp()
        return text_shared, pca_shared, logit_scale_exp, self.logit_bias, aux_pca_pred

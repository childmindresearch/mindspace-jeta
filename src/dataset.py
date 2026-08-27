"""
PyTorch Dataset module for handling paired text embeddings and PCA score matrices.
"""

from typing import Union, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset


class JournalPCADataset(Dataset):
    """PyTorch Dataset holding text embeddings and PCA score matrices.

    Args:
        text_embeddings: Pre-encoded dense text vectors of shape (N, text_dim) as numpy array or tensor.
        pca_scores: 5D (or D-dimensional) PCA component score matrix of shape (N, pca_dim) as numpy array or tensor.
    """

    def __init__(
        self,
        text_embeddings: Union[np.ndarray, torch.Tensor],
        pca_scores: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(text_embeddings, np.ndarray):
            self.text_embeddings = torch.from_numpy(text_embeddings).float()
        else:
            self.text_embeddings = text_embeddings.float()

        if isinstance(pca_scores, np.ndarray):
            self.pca_scores = torch.from_numpy(pca_scores).float()
        else:
            self.pca_scores = pca_scores.float()

        assert len(self.text_embeddings) == len(
            self.pca_scores
        ), f"Mismatch between text embeddings count ({len(self.text_embeddings)}) and PCA scores count ({len(self.pca_scores)})."

    def __len__(self) -> int:
        return len(self.text_embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_embeddings[idx], self.pca_scores[idx]

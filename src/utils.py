"""
Utility functions for text encoding via SentenceTransformers, CSV dataset parsing,
and synthetic dataset generation for testing.
"""

import sys
import os
import random
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch

# Disable pyarrow_hotfix vulnerability patch conflict with modern pyarrow
sys.modules['pyarrow_hotfix'] = type('pyarrow_hotfix', (), {'install': lambda *args, **kwargs: None})()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class TextEncoderWrapper:
    """Wrapper around SentenceTransformer models for extracting dense text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_seq_length: int = 256):
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package is required. Install via `pip install sentence-transformers`."
            )

        print(f"[TextEncoder] Loading model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length
        if hasattr(self.model, "get_embedding_dimension"):
            self._embedding_dim = self.model.get_embedding_dimension()
        else:
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[TextEncoder] Model loaded. Embedding dim: {self._embedding_dim}, max_seq_length: {self.max_seq_length}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        """Encodes a list of raw text strings into a 2D numpy float32 array (N, embedding_dim)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)


def load_csv_dataset(csv_path: str, text_column: str, pca_columns: List[str]) -> Tuple[List[str], np.ndarray]:
    """Reads a CSV dataset and extracts text strings and the PCA component score matrix.

    Args:
        csv_path: Path to the input CSV file.
        text_column: Header name for the free-text column.
        pca_columns: List of header names for the PCA component columns.

    Returns:
        Tuple of (texts list, pca score matrix of shape (N, len(pca_columns))).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")

    missing_pca = [col for col in pca_columns if col not in df.columns]
    if missing_pca:
        raise KeyError(f"PCA columns {missing_pca} not found in CSV. Available columns: {list(df.columns)}")

    # Extract text and fill NAs
    texts = df[text_column].fillna("").astype(str).tolist()

    # Extract PCA component values
    pca_matrix = df[pca_columns].to_numpy(dtype=np.float32)

    return texts, pca_matrix


def generate_synthetic_csv(
    output_csv_path: str,
    text_column: str,
    pca_columns: List[str],
    num_samples: int = 500,
    seed: int = 42,
) -> None:
    """Generates a synthetic CSV file containing realistic journal entries and correlated PCA component scores."""
    if not text_column or not str(text_column).strip():
        raise ValueError("[Data Error] 'text_column' must be explicitly provided as a non-empty string.")

    if not pca_columns:
        raise ValueError("[Data Error] 'pca_columns' must be explicitly provided as a non-empty list of column headers.")

    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    # Sample journal entry text fragments
    topics = [
        "Today was productive. I finished work early and went for a long walk in the park.",
        "Feeling anxious about upcoming deadlines. Thoughts keep intrusive thoughts repeating.",
        "Focused on my main tasks. Completed three major milestones and organized my desk.",
        "Felt overwhelming emotion during the discussion. Need time to reflect and unwind.",
        "Mind wandered frequently today. Struggled to concentrate on my reading assignments.",
        "Great day overall! Reconnected with an old friend and cooked a healthy dinner.",
        "Tired and sluggish. Didn't sleep well last night, so energy was low all afternoon.",
        "Reflecting on personal goals. Excited about starting a new habit routine tomorrow.",
        "Felt a sudden burst of motivation. Cleared my inbox and planned out the entire week.",
        "Stressed about finances and work. Journaling helps me process these heavy feelings.",
    ]

    journal_entries = []
    for _ in range(num_samples):
        # Sample 2-4 sentences to make ~150-300 word journal entry simulation
        sentences = random.choices(topics, k=random.randint(2, 4))
        entry = " ".join(sentences)
        journal_entries.append(entry)

    # Generate normalized PCA components (mean ~ 0, std ~ 1)
    num_components = len(pca_columns)
    pca_data = np.random.randn(num_samples, num_components).astype(np.float32)

    df_dict = {text_column: journal_entries}
    for i, col in enumerate(pca_columns):
        df_dict[col] = pca_data[:, i]

    df = pd.DataFrame(df_dict)
    df.to_csv(output_csv_path, index=False)
    print(f"[Synthetic Data] Generated synthetic dataset with {num_samples} rows at '{output_csv_path}'")

"""
Inference engine and utility functions for projecting raw text and 5D PCA vectors
into the shared metric space and performing cross-modal retrieval.
"""

import os
from typing import List, Dict, Union, Optional
import numpy as np
import torch

from src.config import Config
from src.models import ContrastiveProjectionModel
from src.utils import TextEncoderWrapper


class CLIPPCAPipeline:
    """High-level Pipeline and Inference Engine for the Contrastive Projection Model."""

    def __init__(
        self,
        model: ContrastiveProjectionModel,
        text_encoder: TextEncoderWrapper,
        config: Config,
        device: Optional[str] = None,
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> "CLIPPCAPipeline":
        """Loads a trained pipeline from a model checkpoint path."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint file not found at '{checkpoint_path}'")

        print(f"[InferencePipeline] Loading checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Prioritize architecture and encoder parameters saved inside checkpoint to prevent state_dict mismatch
        ckpt_config = checkpoint.get("config", None)
        if ckpt_config is not None:
            if config is None:
                config = ckpt_config
            else:
                config.model_architecture = ckpt_config.model_architecture
                config.text_encoder = ckpt_config.text_encoder
                config.dataset.pca_columns = ckpt_config.dataset.pca_columns
        elif config is None:
            config = Config()

        arch_config = config.model_architecture
        text_enc_config = config.text_encoder
        pca_input_dim = config.dataset.pca_input_dim

        text_encoder = TextEncoderWrapper(
            model_name=text_enc_config.model_name,
            max_seq_length=text_enc_config.max_seq_length,
        )

        text_input_dim = text_enc_config.text_input_dim or text_encoder.embedding_dim

        model = ContrastiveProjectionModel(
            text_input_dim=text_input_dim,
            pca_input_dim=pca_input_dim,
            shared_dim=arch_config.shared_dim,
            text_head_hidden_dims=arch_config.text_head_hidden_dims,
            text_head_dropout=arch_config.text_head_dropout,
            pca_head_hidden_dims=arch_config.pca_head_hidden_dims,
            initial_temperature=arch_config.initial_temperature,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        print("[InferencePipeline] Checkpoint successfully loaded.")
        return cls(model=model, text_encoder=text_encoder, config=config, device=device)

    def predict_shared_embedding(self, text_list: List[str]) -> np.ndarray:
        """Converts raw text strings to shared space vectors using the text encoder and trained text head."""
        if not text_list:
            return np.empty((0, self.config.model_architecture.shared_dim), dtype=np.float32)

        # 1. Encode text via SentenceTransformer
        dense_embeds = self.text_encoder.encode(text_list, show_progress_bar=False)

        # 2. Pass through trained Text Projection Head & L2 normalize
        with torch.no_grad():
            tensor_in = torch.from_numpy(dense_embeds).float().to(self.device)
            shared_embeds = self.model.encode_text(tensor_in).cpu().numpy()

        return shared_embeds.astype(np.float32)

    def pca_to_shared_embedding(self, pca_scores: Union[List[float], np.ndarray, List[List[float]]]) -> np.ndarray:
        """Converts raw 5D (or D-dimensional) PCA vectors into shared space vectors using the trained PCA head."""
        pca_arr = np.array(pca_scores, dtype=np.float32)

        # Handle single vector input vs batch matrix
        if pca_arr.ndim == 1:
            pca_arr = np.expand_dims(pca_arr, axis=0)

        assert pca_arr.shape[1] == self.config.dataset.pca_input_dim, (
            f"PCA vector dimension ({pca_arr.shape[1]}) does not match model config ({self.config.dataset.pca_input_dim})."
        )

        with torch.no_grad():
            tensor_in = torch.from_numpy(pca_arr).float().to(self.device)
            shared_embeds = self.model.encode_pca(tensor_in).cpu().numpy()

        return shared_embeds.astype(np.float32)

    def cross_modal_retrieval(
        self,
        query_pca_vector: Union[List[float], np.ndarray],
        text_database: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Given an arbitrary target 5D PCA profile vector, finds and ranks the top K journal entries

        in the database with the highest cosine similarity in the shared metric space.

        Args:
            query_pca_vector: Target 5D PCA profile (e.g., [1.5, -1.0, 0.5, 0.0, 0.5])
            text_database: List of free-text journal entry strings.
            top_k: Number of top results to return.

        Returns:
            Ranked list of dictionaries containing rank, text entry, similarity_score, and index.
        """
        if not text_database:
            return []

        # Project 5D PCA query to shared space
        query_shared = self.pca_to_shared_embedding(query_pca_vector)

        # Project text database entries to shared space
        text_shared_db = self.predict_shared_embedding(text_database)

        # Compute cosine similarity
        cosine_sims = np.dot(text_shared_db, query_shared.T).squeeze(axis=-1)  # (N,)

        # Rank top_k indices descending
        top_k = min(top_k, len(text_database))
        top_indices = np.argsort(-cosine_sims)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "similarity_score": float(cosine_sims[idx]),
                    "text": text_database[idx],
                }
            )

        return results

    def predict_pca_components(
        self,
        text_list: List[str],
        reference_pca_matrix: Optional[np.ndarray] = None,
        reference_texts: Optional[List[str]] = None,
        temperature: float = 10.0,
        use_direct_head: bool = True,
    ) -> np.ndarray:
        """Maps 1...N raw free-text journal entries to predicted 5D PCA component score space.

        Uses direct trained 5D linear decoder head or kernel-weighted similarity interpolation over exemplars.

        Args:
            text_list: List of N raw text strings.
            reference_pca_matrix: Optional reference dataset 5D PCA score matrix of shape (M, 5).
            reference_texts: Optional list of M reference text entries.
            temperature: Softmax temperature scaling factor for similarity weighting.
            use_direct_head: If True, uses the trained direct 5D PCA decoder head.

        Returns:
            Predicted PCA component matrix of shape (N, 5).
        """
        if not text_list:
            return np.empty((0, self.config.dataset.pca_input_dim), dtype=np.float32)

        # 1. Project input text entries to shared space (N, shared_dim)
        text_shared = self.predict_shared_embedding(text_list)

        # 2. Direct head prediction using trained 5D PCA decoder
        if use_direct_head and hasattr(self.model, "pca_decoder"):
            with torch.no_grad():
                tensor_in = torch.from_numpy(text_shared).float().to(self.device)
                predicted_pca = self.model.predict_pca(tensor_in).cpu().numpy()
            return predicted_pca.astype(np.float32)

        # 3. Softmax temperature-weighted interpolation over reference exemplars
        if reference_pca_matrix is not None:
            ref_pca_shared = self.pca_to_shared_embedding(reference_pca_matrix)
            sim_matrix = np.dot(text_shared, ref_pca_shared.T)
            exp_sims = np.exp(temperature * (sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)))
            attn_weights = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)  # (N, M)
            predicted_pca = np.dot(attn_weights, reference_pca_matrix)  # (N, 5)
            return predicted_pca.astype(np.float32)

        raise ValueError("[Inference Error] Must provide reference_pca_matrix or enable use_direct_head.")


# Aliases for MindSpace-JETA pipeline naming convention
JETAPCAPipeline = CLIPPCAPipeline
CLAPPCAPipeline = CLIPPCAPipeline



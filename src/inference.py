"""
Inference engine and utility functions for projecting raw text and 5D PCA vectors
into the shared 16D metric space and performing cross-modal retrieval.
"""

import os
from typing import List, Dict, Union, Tuple, Optional
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

        if config is None:
            config = checkpoint.get("config", Config())

        text_encoder = TextEncoderWrapper(
            model_name=config.text_encoder.model_name,
            max_seq_length=config.text_encoder.max_seq_length,
        )

        text_input_dim = config.text_encoder.text_input_dim or text_encoder.embedding_dim

        model = ContrastiveProjectionModel(
            text_input_dim=text_input_dim,
            pca_input_dim=config.dataset.pca_input_dim,
            shared_dim=config.model_architecture.shared_dim,
            text_head_hidden_dims=config.model_architecture.text_head_hidden_dims,
            text_head_dropout=config.model_architecture.text_head_dropout,
            pca_head_hidden_dims=config.model_architecture.pca_head_hidden_dims,
            initial_temperature=config.model_architecture.initial_temperature,
            use_rff_expansion=getattr(config.model_architecture, "use_rff_expansion", True),
            rff_dim=getattr(config.model_architecture, "rff_dim", 128),
            use_swiglu_residual=getattr(config.model_architecture, "use_swiglu_residual", True),
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
        """Converts raw 5D PCA vectors into shared space vectors using the trained PCA head."""
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
            query_pca_vector: Target 5D PCA profile (e.g., [1.5, -2.0, 0.5, 0.0, 1.0])
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
        reference_pca_matrix: np.ndarray,
        reference_texts: Optional[List[str]] = None,
        temperature: float = 10.0,
    ) -> np.ndarray:
        """Maps 1...N raw free-text journal entries to predicted 5D PCA component score space.

        Combines direct linear auxiliary decoding from the text head with kernel-weighted similarity interpolation.

        Args:
            text_list: List of N raw text strings.
            reference_pca_matrix: Reference dataset 5D PCA score matrix of shape (M, 5).
            reference_texts: Optional list of M reference text entries.
            temperature: Softmax temperature scaling factor for similarity weighting.

        Returns:
            Predicted PCA component matrix of shape (N, 5).
        """
        if not text_list:
            return np.empty((0, self.config.dataset.pca_input_dim), dtype=np.float32)

        # 1. Direct auxiliary prediction from text head
        dense_embeds = self.text_encoder.encode(text_list, show_progress_bar=False)
        with torch.no_grad():
            tensor_in = torch.from_numpy(dense_embeds).float().to(self.device)
            direct_pca_pred = self.model.predict_pca_from_text(tensor_in).cpu().numpy()

        # 2. Kernel-weighted similarity interpolation over reference exemplars
        text_shared = self.predict_shared_embedding(text_list)
        ref_pca_shared = self.pca_to_shared_embedding(reference_pca_matrix)

        sim_matrix = np.dot(text_shared, ref_pca_shared.T)
        exp_sims = np.exp(temperature * (sim_matrix - np.max(sim_matrix, axis=1, keepdims=True)))
        attn_weights = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

        kernel_pca_pred = np.dot(attn_weights, reference_pca_matrix)

        # Combine direct aux prediction with kernel interpolation
        final_predicted_pca = 0.5 * direct_pca_pred + 0.5 * kernel_pca_pred
        return final_predicted_pca.astype(np.float32)

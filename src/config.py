"""
Configuration loader and schema definition module.
Handles reading YAML configuration files and validating parameters.
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DatasetConfig:
    train_csv_path: str = "./data/train_data.csv"
    inference_csv_path: str = "./data/eval_data.csv"
    text_column: str = "journal_entry"
    pca_columns: List[str] = field(default_factory=lambda: ["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5"])
    sample_query_pca: List[float] = field(default_factory=lambda: [1.5, -1.0, 0.5, 0.0, 0.5])

    @property
    def pca_input_dim(self) -> int:
        return len(self.pca_columns)


@dataclass
class TextEncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 256
    text_input_dim: Optional[int] = None  # Auto-detected if None


@dataclass
class ModelArchConfig:
    shared_dim: int = 16
    text_head_hidden_dims: List[int] = field(default_factory=lambda: [128])
    text_head_dropout: float = 0.3
    pca_head_hidden_dims: List[int] = field(default_factory=lambda: [32])
    initial_temperature: float = 0.07
    use_rff_expansion: bool = True
    rff_dim: int = 128
    use_swiglu_residual: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 25
    train_split: float = 0.8
    seed: int = 42
    loss_type: str = "siglip"
    auxiliary_loss_weight: float = 0.5
    soft_target_sigma: float = 1.0


@dataclass
class PathsConfig:
    output_dir: str = "./outputs"
    logs_dir: str = "./outputs/logs"
    model_checkpoint: str = "./outputs/contrastive_model.pt"
    embeddings_output: str = "./outputs/projected_embeddings.pt"


@dataclass
class OptunaConfig:
    n_trials: int = 25
    timeout: Optional[int] = None
    best_config_path: str = "./config.yaml"
    archive_dir: str = "./config_archive"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    model_architecture: ModelArchConfig = field(default_factory=ModelArchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)

    def validate(self) -> None:
        """Validates configuration parameters and raises descriptive ValueError if invalid."""
        if not self.dataset.text_column or not str(self.dataset.text_column).strip():
            raise ValueError("[Configuration Error] 'dataset.text_column' must be defined as a non-empty string in config.yaml.")

        if not self.dataset.pca_columns:
            raise ValueError("[Configuration Error] 'dataset.pca_columns' must be defined as a non-empty list of column headers in config.yaml.")

        if not self.model_architecture.text_head_hidden_dims:
            raise ValueError("[Configuration Error] 'model_architecture.text_head_hidden_dims' must be defined as a non-empty list in config.yaml.")

        if not self.model_architecture.pca_head_hidden_dims:
            raise ValueError("[Configuration Error] 'model_architecture.pca_head_hidden_dims' must be defined as a non-empty list in config.yaml.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build Config object from nested dictionary and validate schema."""
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        text_cfg = TextEncoderConfig(**data.get("text_encoder", {}))
        model_cfg = ModelArchConfig(**data.get("model_architecture", {}))
        train_cfg = TrainingConfig(**data.get("training", {}))
        paths_cfg = PathsConfig(**data.get("paths", {}))
        optuna_cfg = OptunaConfig(**data.get("optuna", {}))
        cfg = cls(
            dataset=dataset_cfg,
            text_encoder=text_cfg,
            model_architecture=model_cfg,
            training=train_cfg,
            paths=paths_cfg,
            optuna=optuna_cfg,
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config dataclass to nested dictionary for YAML export."""
        return {
            "dataset": {
                "train_csv_path": self.dataset.train_csv_path,
                "inference_csv_path": self.dataset.inference_csv_path,
                "text_column": self.dataset.text_column,
                "pca_columns": self.dataset.pca_columns,
                "sample_query_pca": self.dataset.sample_query_pca,
            },
            "text_encoder": {
                "model_name": self.text_encoder.model_name,
                "max_seq_length": self.text_encoder.max_seq_length,
                "text_input_dim": self.text_encoder.text_input_dim,
            },
            "model_architecture": {
                "shared_dim": self.model_architecture.shared_dim,
                "text_head_hidden_dims": self.model_architecture.text_head_hidden_dims,
                "text_head_dropout": self.model_architecture.text_head_dropout,
                "pca_head_hidden_dims": self.model_architecture.pca_head_hidden_dims,
                "initial_temperature": self.model_architecture.initial_temperature,
                "use_rff_expansion": self.model_architecture.use_rff_expansion,
                "rff_dim": self.model_architecture.rff_dim,
                "use_swiglu_residual": self.model_architecture.use_swiglu_residual,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "epochs": self.training.epochs,
                "train_split": self.training.train_split,
                "seed": self.training.seed,
                "loss_type": self.training.loss_type,
                "auxiliary_loss_weight": self.training.auxiliary_loss_weight,
                "soft_target_sigma": self.training.soft_target_sigma,
            },
            "paths": {
                "output_dir": self.paths.output_dir,
                "logs_dir": self.paths.logs_dir,
                "model_checkpoint": self.paths.model_checkpoint,
                "embeddings_output": self.paths.embeddings_output,
            },
            "optuna": {
                "n_trials": self.optuna.n_trials,
                "timeout": self.optuna.timeout,
                "best_config_path": self.optuna.best_config_path,
                "archive_dir": self.optuna.archive_dir,
            },
        }


def save_config(config: Config, output_path: str) -> None:
    """Save Config object to a YAML file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    print(f"[Config] Saved configuration to '{output_path}'")


def archive_existing_config(config_path: str = "config.yaml", archive_dir: str = "./config_archive") -> Optional[str]:
    """Archives the existing configuration file into archive_dir with a timestamp identifier.

    Returns the path to the archived file, or None if config_path does not exist.
    """
    if not os.path.exists(config_path):
        return None

    os.makedirs(archive_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f"config_{timestamp}.yaml"
    archive_path = os.path.join(archive_dir, archive_filename)

    shutil.copy2(config_path, archive_path)
    print(f"[Config Archive] Archived existing '{config_path}' to '{archive_path}'")
    return archive_path


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file. If not found, returns default Config."""
    if not os.path.exists(config_path):
        print(f"[Warning] Config file '{config_path}' not found. Using default configurations.")
        return Config()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config.from_dict(data)


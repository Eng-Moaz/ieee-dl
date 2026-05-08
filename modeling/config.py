"""
Centralized configuration for the Facial Emotion Recognition pipeline.

All hyperparameters, data-split ratios, and filesystem paths are defined here
as a single frozen dataclass instance (``config``).  Importing this module in
any other component guarantees a consistent, read-only view of the project
settings.
"""

import os
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Immutable configuration container for the FER training pipeline.

    Using a frozen dataclass prevents accidental mutation of shared settings
    across modules.  Computed paths are exposed as ``@property`` attributes so
    that directories are created on first access rather than at import time.
    """

    # --- Environment ---
    is_kaggle: bool = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    # --- Model architecture ---
    num_classes: int   = 7
    dropout:     float = 0.4
    img_size:    int   = 224

    # --- Phase 1: head-only training ---
    phase1_epochs: int   = 50
    phase1_lr:     float = 1e-3

    # --- Phase 2: full fine-tuning ---
    phase2_epochs: int   = 50
    phase2_lr:     float = 1e-5

    # --- General training ---
    batch_size:              int   = 64
    num_workers:             int   = 12
    focal_gamma:             float = 2.0
    lr_patience:             int   = 3
    early_stopping_patience: int   = 7

    # --- Stratified data split ---
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15
    split_seed:  int   = 42

    # --- Emotion class labels (must match ImageFolder directory order) ---
    emotion_labels: tuple = (
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    )

    @property
    def raw_data_dir(self) -> Path:
        """Absolute path to the ImageFolder-compatible dataset root.

        Resolves to the Kaggle kernel input path when executed on Kaggle,
        otherwise falls back to the local ``data/processed_data`` directory.
        """
        if self.is_kaggle:
            return Path('/kaggle/input/datasets/fahadullaha/facial-emotion-recognition-dataset/processed_data')
        return Path(__file__).parent.parent / 'data' / 'processed_data'

    @property
    def checkpoint_dir(self) -> Path:
        """Absolute path to the checkpoint directory.

        The directory is created on first access if it does not already exist.
        """
        path = Path(__file__).parent.parent / 'checkpoints'
        path.mkdir(exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """Absolute path to the TensorBoard log and evaluation output directory.

        The directory is created on first access if it does not already exist.
        """
        path = Path(__file__).parent.parent / 'logs'
        path.mkdir(exist_ok=True)
        return path

    @property
    def device(self) -> str:
        """Training device: ``'cuda'`` if a GPU is available, otherwise ``'cpu'``."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()


if __name__ == "__main__":
    print(f"Running on        : {'Kaggle' if config.is_kaggle else 'Local'}")
    print(f"Device            : {config.device}")
    print(f"Raw data          : {config.raw_data_dir}")
    print(f"Batch size        : {config.batch_size}")
    print(f"Split ratio       : {config.train_ratio} / {config.val_ratio} / {config.test_ratio}")
    print(f"Phase 1 LR        : {config.phase1_lr}  for {config.phase1_epochs} epochs")
    print(f"Phase 2 LR        : {config.phase2_lr}  for {config.phase2_epochs} epochs")
    print(f"Emotions          : {config.emotion_labels}")
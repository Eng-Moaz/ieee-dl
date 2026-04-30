import os
import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:

    # Environment
    is_kaggle: bool = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    # Model
    num_classes: int   = 7
    dropout:     float = 0.4
    img_size:    int   = 224

    # Phase 1
    phase1_epochs: int   = 15
    phase1_lr:     float = 1e-3

    # Phase 2
    phase2_epochs: int   = 15
    phase2_lr:     float = 1e-5


    # General training
    batch_size:   int   = 32
    num_workers:  int   = 2
    lr_step_size: int   = 10
    lr_gamma:     float = 0.5

    # Data split
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15
    split_seed:  int   = 42

    # Labels
    emotion_labels: tuple = (
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    )

    # Paths
    @property
    def raw_data_dir(self) -> Path:
        if self.is_kaggle:
            return Path('/kaggle/input/datasets/fahadullaha/facial-emotion-recognition-dataset/processed_data')
        return Path(__file__).parent.parent / 'data' 


    @property
    def checkpoint_dir(self) -> Path:
        path = Path(__file__).parent.parent / 'checkpoints'
        path.mkdir(exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        path = Path(__file__).parent.parent / 'logs'
        path.mkdir(exist_ok=True)
        return path

    # Evaluation results directory
    @property
    def eval_results_dir(self) -> Path:
        path = self.log_dir / 'evaluation'
        path.mkdir(exist_ok=True)
        return path

    # Number of target classes
    @property
    def num_classes(self) -> int:
        return len(self.emotion_labels)
    @property
    def device(self) -> str:
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
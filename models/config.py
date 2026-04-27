import os
import torch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:

    # ── Environment ───────────────────────────────────────────────────────────
    is_kaggle: bool = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes: int   = 7        # angry, disgust, fear, happy, neutral, sad, surprise
    dropout:     float = 0.4      # dropout probability in the classification head
    img_size:    int   = 224      # ResNet-50 expects 224×224

    # ── Training — Phase 1 (head only, backbone frozen) ───────────────────────
    phase1_epochs: int   = 15
    phase1_lr:     float = 1e-3   # higher LR is fine since only the head trains

    # ── Training — Phase 2 (full model fine-tuning) ───────────────────────────
    phase2_epochs: int   = 15
    phase2_lr:     float = 1e-5   # must be small — backbone weights are already good

    # ── Training — General ────────────────────────────────────────────────────
    batch_size:   int   = 32
    num_workers:  int   = 2       # parallel data loading workers (set 0 on Windows if errors)
    lr_step_size: int   = 10      # StepLR: drop LR every N epochs
    lr_gamma:     float = 0.5     # StepLR: multiply LR by this factor each step

    # ── Emotion class labels (in the order your dataset folders are sorted) ───
    emotion_labels: tuple = (
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    )

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def data_root(self) -> Path:
        if self.is_kaggle:
            return Path('/kaggle/input/datasets/fahadullaha/facial-emotion-recognition-dataset/processed_data')
        return Path(__file__).parent / 'data'

    @property
    def train_dir(self) -> Path:
        return self.data_root / 'train'

    @property
    def val_dir(self) -> Path:
        return self.data_root / 'val'

    @property
    def test_dir(self) -> Path:
        return self.data_root / 'test'

    @property
    def checkpoint_dir(self) -> Path:
        path = Path(__file__).parent / 'checkpoints'
        path.mkdir(exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        path = Path(__file__).parent / 'runs'
        path.mkdir(exist_ok=True)
        return path

    @property
    def device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Singleton — import this everywhere ───────────────────────────────────────
config = Config()


# ── Quick check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running on   : {'Kaggle' if config.is_kaggle else 'Local'}")
    print(f"Device       : {config.device}")
    print(f"Data root    : {config.data_root}")
    print(f"Train dir    : {config.train_dir}")
    print(f"Val dir      : {config.val_dir}")
    print(f"Batch size   : {config.batch_size}")
    print(f"Phase 1 LR   : {config.phase1_lr}  for {config.phase1_epochs} epochs")
    print(f"Phase 2 LR   : {config.phase2_lr}  for {config.phase2_epochs} epochs")
    print(f"Emotions     : {config.emotion_labels}")
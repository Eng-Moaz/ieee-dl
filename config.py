import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:

    # Environment State
    is_kaggle: bool = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    # Data Path
    @property
    def data_root(self):
        if self.is_kaggle:
            return Path('/kaggle/input/datasets/fahadullaha/facial-emotion-recognition-dataset/processed_data')
        return Path(__file__).parent / 'data' / 'processed_data'

config = Config()
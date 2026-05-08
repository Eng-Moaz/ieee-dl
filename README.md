# Facial Emotion Recognition (FER)

A modular deep learning pipeline for classifying facial expressions into seven emotion categories using a fine-tuned ResNet-50 backbone trained on the FER-2013 dataset.

---

## Overview

This project implements a two-phase transfer learning strategy to address the inherent class imbalance in the FER-2013 dataset:

1. **Phase 1 — Head Training**: The ResNet-50 backbone is frozen and only the classification head is trained at a higher learning rate.
2. **Phase 2 — Fine-tuning**: The full network is unfrozen and trained end-to-end at a significantly lower learning rate.

Focal Loss with inverse-frequency class weighting is used throughout both phases to penalize errors on under-represented emotion classes.

---

## Emotion Classes

| Index | Label    |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Neutral  |
| 5     | Sad      |
| 6     | Surprise |

---

## Project Structure

```
ieee-dl/
├── modeling/
│   ├── config.py          # Centralized hyperparameter and path configuration
│   ├── model.py           # FERModel definition (ResNet-50 backbone + classification head)
│   ├── losses.py          # Focal Loss implementation with optional class weights
│   └── train.py           # Two-phase training loop with early stopping and TensorBoard logging
├── preprocessing/
│   └── data_loaders.py    # Stratified train/val/test split and DataLoader construction
├── utils/
│   └── evaluation.py      # Post-training evaluation: confusion matrix, ROC curve, classification report
├── checkpoints/           # Saved model weights (best_model.pth)
├── logs/                  # TensorBoard event files and evaluation outputs
├── data/
│   └── processed_data/    # Expected ImageFolder-compatible dataset directory
└── pyproject.toml
```

---

## Setup

**Requirements:** Python ≥ 3.13, CUDA-capable GPU recommended.

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

---

## Dataset

The pipeline expects the dataset to be organized in [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format under `data/processed_data/`:

```
data/processed_data/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

When running on Kaggle, the dataset path is automatically resolved to the appropriate kernel input directory.

---

## Training

```bash
python -m modeling.train
```

Training proceeds in two phases as configured in `modeling/config.py`. Checkpoints are saved to `checkpoints/best_model.pth` whenever validation accuracy improves. Early stopping is applied if no improvement is observed for `early_stopping_patience` consecutive epochs.

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir logs/
```

---

## Evaluation

```bash
python -m utils.evaluation
```

Loads the best checkpoint and runs inference on the held-out test set, producing:

- `logs/confusion_matrix.png`
- `logs/roc_curve.png`
- `logs/report.txt` — per-class precision, recall, and F1 scores

---

## Configuration

All hyperparameters are defined in `modeling/config.py` as a frozen dataclass. Key parameters:

| Parameter                  | Default | Description                                      |
|----------------------------|---------|--------------------------------------------------|
| `num_classes`              | 7       | Number of emotion categories                     |
| `dropout`                  | 0.4     | Dropout rate in the classification head          |
| `batch_size`               | 64      | Samples per mini-batch                           |
| `phase1_epochs`            | 50      | Maximum epochs for head-only training            |
| `phase1_lr`                | 1e-3    | Learning rate for Phase 1                        |
| `phase2_epochs`            | 50      | Maximum epochs for full fine-tuning              |
| `phase2_lr`                | 1e-5    | Learning rate for Phase 2                        |
| `focal_gamma`              | 2.0     | Focusing exponent for Focal Loss                 |
| `lr_patience`              | 3       | Epochs before LR is reduced on plateau           |
| `early_stopping_patience`  | 7       | Epochs without improvement before stopping       |
| `train_ratio`              | 0.70    | Fraction of data used for training               |
| `val_ratio`                | 0.15    | Fraction of data used for validation             |
| `test_ratio`               | 0.15    | Fraction of data used for testing                |

"""
Post-training evaluation for the Facial Emotion Recognition pipeline.

Loads the best saved checkpoint, runs inference on the held-out test set,
and produces three evaluation artefacts saved to the configured log directory:

- ``confusion_matrix.png`` — normalized confusion matrix heatmap.
- ``roc_curve.png``        — per-class one-vs-rest ROC curves with AUC values.
- ``report.txt``           — full per-class precision, recall, and F1 scores.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from modeling.config import config
from modeling.model import FERModel
from preprocessing.data_loaders import get_data_loaders

DEVICE      = config.device
CLASSES     = config.emotion_labels
NUM_CLASSES = config.num_classes
OUTPUT_DIR  = config.log_dir


def run_evaluation(model, data_loader):
    """Run inference over ``data_loader`` and collect predictions and probabilities.

    Args:
        model: A trained :class:`~modeling.model.FERModel` instance.
        data_loader: DataLoader wrapping the evaluation dataset.

    Returns:
        Tuple of NumPy arrays ``(all_labels, all_preds, all_probs)`` where
        ``all_probs`` contains softmax-normalised class probabilities.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    print(f"Starting evaluation on {DEVICE}...")
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_all(y_true, y_pred, y_probs):
    """Generate and save all evaluation plots and the classification report.

    Produces:
        - Confusion matrix heatmap (``confusion_matrix.png``).
        - Multi-class ROC curve with per-class AUC (``roc_curve.png``).
        - Text-format classification report (``report.txt``).

    Args:
        y_true:  Ground-truth class indices, shape ``(N,)``.
        y_pred:  Predicted class indices, shape ``(N,)``.
        y_probs: Softmax class probabilities, shape ``(N, num_classes)``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    plt.figure(figsize=(12, 10))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'ROC {CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC Curve')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write(report)
    print("\n" + report)


def main():
    """Entry point: load the best checkpoint and run the full evaluation pipeline."""
    checkpoint_path = config.checkpoint_dir / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading FERModel weights from: {checkpoint_path}")
    model = FERModel().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

    _, _, test_loader = get_data_loaders(config.raw_data_dir, config.batch_size)

    if test_loader is None:
        print("DataLoader not ready. Aborting.")
        return

    print("Running inference...")
    y_true, y_pred, y_probs = run_evaluation(model, test_loader)

    print("Generating evaluation plots...")
    plot_all(y_true, y_pred, y_probs)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
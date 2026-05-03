import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Project Modules Import
from modeling.config import config
from modeling.model import FERModel
from preprocessing.data_loaders import get_data_loaders

# 2. Global Constants from Config
DEVICE = config.device
CLASSES = config.emotion_labels
NUM_CLASSES = config.num_classes
OUTPUT_DIR = config.log_dir

# --- FUNCTIONS ---

def run_evaluation(model, data_loader):
    """Execute model inference and collect raw results"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    print(f"[*] Starting evaluation on {DEVICE}...")
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_all(y_true, y_pred, y_probs):
    """Generate and save evaluation metrics and plots"""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
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

    # 3. Save Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write(report)
    print("\n" + report)

# --- MAIN EXECUTION ---

def main():
    # 1. Load trained model weights
    checkpoint_path = config.checkpoint_dir / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"[!] Error: Checkpoint missing at {checkpoint_path}")
        return

    print(f"[*] Loading FERModel weights from: {checkpoint_path}")
    model = FERModel().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

    # 2. Initialize real data loader
    _, _, test_loader = get_data_loaders(config.raw_data_dir, config.batch_size)
    
    if test_loader is None:
        print("[!] Execution stopped: DataLoader not ready.")
        return

    # 3. Step-by-step evaluation pipeline
    print("[*] Running inference...")
    y_true, y_pred, y_probs = run_evaluation(model, test_loader)

    # 4. Generate results and save to the files
    print("[*] Visualizing results...")
    plot_all(y_true, y_pred, y_probs)

    print(f"\n[DONE] All outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
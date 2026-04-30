# import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
import os

#import model
try:
    from model import FERModel
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("[Warning] model.py not found. Using dummy model for testing.")

#environment setup 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_mock_data():
    """توليد بيانات وهمية للتجربة قبل وصول الداتا الحقيقية"""
    print("[Info] Generating MOCK data for testing...")
    images = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 7, (100,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=20)

def run_evaluation(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

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
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
    
    # 2. ROC Curve (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(7))
    plt.figure(figsize=(10, 8))
    for i in range(7):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'ROC {CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC Curve')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")

    # 3. Classification Report (Text)
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
        f.write(report)
    print("\n" + report)

def main():
    # تحميل الموديل (حقيقي أو تجريبي)
    if HAS_MODEL:
        model = FERModel().to(DEVICE)
        # لو عندك ملف أوزان فك التشفير عن السطر اللي جاي:
        # model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    else:
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*224*224, 7)).to(DEVICE)

    # تحميل الداتا (حالياً Mock)
    test_loader = get_mock_data()

    # تنفيذ التقييم
    y_true, y_pred, y_probs = run_evaluation(model, test_loader)

    # رسم النتائج
    plot_all(y_true, y_pred, y_probs)
    print(f"\nDONE! All results saved in folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
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

#import  config
from modeling.config import config

#import models
from modeling.model import FERModel

# setup 
DEVICE = config.device
CLASSES = config.emotion_labels
NUM_CLASSES = config.num_classes
OUTPUT_DIR = config.eval_results_dir 

from torchvision import datasets, transforms

def get_real_test_loader():
    # Define image transformations
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Set path from config
    test_path = config.raw_data_dir / 'test'
    
    if not test_path.exists():
        print(f"[!] Path not found: {test_path}")
        return None

    # Load dataset using ImageFolder
    test_dataset = datasets.ImageFolder(root=str(test_path), transform=test_transforms)
    
    # Return real DataLoader
    return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

def run_evaluation(model, data_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    print(f"[*] Evaluating on {config.device}...")
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to config device
            images, labels = images.to(config.device), labels.to(config.device)
            
            # Model inference
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_all(y_true, y_pred, y_probs):
    # 1. Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(config.eval_results_dir / "confusion_matrix.png")
    
    # 2. Plot ROC Curve (One-vs-Rest)
    # Use config.num_classes for dynamic range
    y_true_bin = label_binarize(y_true, classes=range(config.num_classes))
    plt.figure(figsize=(12, 10))
    for i in range(config.num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, label=f'ROC {CLASSES[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC Curve')
    plt.legend()
    plt.savefig(config.eval_results_dir / "roc_curve.png")

    # 3. Save Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    with open(config.eval_results_dir / "report.txt", "w") as f:
        f.write(report)
    print("\n" + report)
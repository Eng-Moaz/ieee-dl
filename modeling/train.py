import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from modeling.config import config
from modeling.losses import FocalLoss


def train(
    model,
    train_loader,
    val_loader,
    epochs=None,
    lr=None,
):
    save_path = str(config.checkpoint_dir / "best_model.pth")

    device = config.device
    print(f"Training on: {device}")

    model = model.to(device)

    print("Calculating class weights for Focal Loss...")
    if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset, 'indices'):
        all_targets = train_loader.dataset.dataset.targets
        labels = [all_targets[i] for i in train_loader.dataset.indices]
    else:
        labels = [s[1] for s in train_loader.dataset.samples]
        
    class_counts = np.bincount(labels, minlength=config.num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights * config.num_classes / np.sum(weights)
    alpha = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=alpha, gamma=config.focal_gamma)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.lr_patience, factor=0.5)

    log_dir = str(config.log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs}] Train", leave=False)
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            
            train_pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / train_total
        avg_train_acc  = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{epochs}] Val  ", leave=False)
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / val_total
        avg_val_acc  = val_correct / val_total

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch:02d}/{epochs}]  "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {avg_train_acc*100:.1f}%  |  "
            f"Val Loss: {avg_val_loss:.4f}  Acc: {avg_val_acc*100:.1f}%  |  "
            f"LR: {current_lr:.0e}"
        )

        writer.add_scalars("Loss", {"train": avg_train_loss, "val": avg_val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": avg_train_acc, "val": avg_val_acc}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved ({avg_val_acc*100:.1f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered: No improvement in validation accuracy for {config.early_stopping_patience} epochs.")
                break

    writer.close()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"Weights saved to: {save_path}")
    print(f"View logs with:   tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    from modeling.model import FERModel
    from preprocessing.data_loaders import get_data_loaders

    model = FERModel()

    train_loader, val_loader, _ = get_data_loaders(config.raw_data_dir, config.batch_size)

    # Phase 1: train only the head (backbone frozen)
    print("--- Phase 1: Training Head ---")
    train(model, train_loader, val_loader, epochs=config.phase1_epochs, lr=config.phase1_lr)

    # Phase 2: unfreeze backbone and fine-tune everything with a tiny LR
    print("--- Phase 2: Fine-tuning Backbone ---")
    model.unfreeze_backbone()
    train(model, train_loader, val_loader, epochs=config.phase2_epochs, lr=config.phase2_lr)

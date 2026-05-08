"""
DataLoader construction for the Facial Emotion Recognition pipeline.

A single :func:`get_data_loaders` call performs a stratified three-way split
of the dataset and returns ready-to-use PyTorch DataLoaders for training,
validation, and testing.  Training data receives augmentation transforms;
validation and test data receive only normalization.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from modeling.config import config


def get_data_loaders(data_dir, batch_size=32):
    """Build stratified train, validation, and test DataLoaders from an ImageFolder root.

    The dataset is split according to the ratios defined in
    :data:`~modeling.config.config` using two successive stratified splits to
    preserve class distribution across all three subsets.

    Two separate :class:`~torchvision.datasets.ImageFolder` instances are
    created from the same root so that augmentation transforms are applied
    exclusively to training samples.

    Args:
        data_dir: Path-like object pointing to the ImageFolder-compatible
            dataset root directory.
        batch_size: Number of samples per mini-batch.  Defaults to ``32``.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset twice so each split can use its own transform pipeline.
    train_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    eval_full_dataset  = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    total_size = len(train_full_dataset)
    targets = train_full_dataset.targets

    test_val_ratio = config.val_ratio + config.test_ratio
    test_ratio_relative = config.test_ratio / test_val_ratio

    # First split: train vs. (val + test)
    train_indices, temp_indices = train_test_split(
        range(total_size),
        test_size=test_val_ratio,
        random_state=config.split_seed,
        stratify=targets
    )

    # Second split: val vs. test, stratified on the temporary subset's labels.
    temp_targets = [targets[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio_relative,
        random_state=config.split_seed,
        stratify=temp_targets
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset   = Subset(eval_full_dataset,  val_indices)
    test_dataset  = Subset(eval_full_dataset,  test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=config.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader
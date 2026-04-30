import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from modeling.config import config

def get_data_loaders(data_dir, batch_size=32):
    """ create train, val, and test loaders using ImageFolder """
    
    # 1. Data Transforms (Train gets augmentation, Val/Test get only resize/norm)
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

    # 2. Load dataset multiple times to apply different transforms
    train_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    eval_full_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transforms)

    # 3. Split data into Train (70%), Val (15%), Test (15%) using Stratified Splitting
    total_size = len(train_full_dataset)
    targets = train_full_dataset.targets
    
    # Calculate proportions
    test_val_ratio = config.val_ratio + config.test_ratio
    test_ratio_relative = config.test_ratio / test_val_ratio

    # First split: Train vs Temp (Val + Test)
    train_indices, temp_indices = train_test_split(
        range(total_size),
        test_size=test_val_ratio,
        random_state=config.split_seed,
        stratify=targets
    )

    # Get targets for the temp split to stratify again
    temp_targets = [targets[i] for i in temp_indices]

    # Second split: Val vs Test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio_relative,
        random_state=config.split_seed,
        stratify=temp_targets
    )

    # Apply indices to the datasets with correct transforms
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(eval_full_dataset, val_indices)
    test_dataset = Subset(eval_full_dataset, test_indices)


    # 4. Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

def prepare_data(data_dir, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Split train_data into train and validation sets first
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Calculate class weights for balanced sampling - only for the training subset
    # Get indices from the subset
    train_indices = train_data.indices
    # Get original labels
    original_targets = [train_data.dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(original_targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[original_targets]
    
    # Create weighted sampler for the training set
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Use the sampler for training data
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():
    data_dir = os.path.join('..', 'dataset')
    batch_size = 32
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    print('Number of training images:', len(train_loader.dataset))
    print('Number of validation images:', len(val_loader.dataset))
    print('Number of test images:', len(test_loader.dataset))
    
    # Print class distribution in training set
    class_counts = {}
    for _, label in train_loader.dataset:
        if hasattr(train_loader.dataset, 'dataset'):
            # If using a Subset from random_split
            class_name = train_loader.dataset.dataset.classes[label]
        else:
            class_name = train_loader.dataset.classes[label]
        
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("Class distribution in training set:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(images[0].shape, labels[0].item())
        break

if __name__ == '__main__':
    main()
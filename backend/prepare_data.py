import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

def prepare_data(data_dir, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Split train_data into train and validation sets
    val_size = int(len(train_data) * val_split)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    class_counts = [sum(1 for _, y in train_data.samples if y == i) for i in range(len(train_data.classes))]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[y] for _, y in train_data.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():
    data_dir = os.path.join('..', 'dataset')
    batch_size = 32
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    print('Number of training images:', len(train_loader.dataset))
    print('Number of validation images:', len(val_loader.dataset))
    print('Number of test images:', len(test_loader.dataset))

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(images[0].shape, labels[0].item())
        break

if __name__ == '__main__':
    main()
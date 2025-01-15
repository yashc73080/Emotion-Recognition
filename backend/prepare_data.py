import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    data_dir = os.path.join('..', 'dataset')
    batch_size = 32
    train_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    print('Number of training images:', len(train_loader.dataset))
    print('Number of test images:', len(test_loader.dataset))

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        print(images[0].shape, labels[0].item())
        break

if __name__ == '__main__':
    main()
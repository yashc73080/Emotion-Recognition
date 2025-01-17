import os
import sys
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join('..')))

from prepare_data import prepare_data

class EmotionRecognitionCNN(nn.Module):
    def __init__(self):
        super(EmotionRecognitionCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 6 * 6)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
    
    def train_model(self, train_loader, lr=0.001, num_epochs=5, device='cpu'):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        error = nn.CrossEntropyLoss()
        self.train()

        for epoch in range(num_epochs):
            correct = 0
            total_loss = 0

            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                X_train, y_train = X_train.to(device), y_train.to(device)
                
                optimizer.zero_grad()
                output = self.forward(X_train)
                loss = error(output, y_train)
                loss.backward()
                optimizer.step()

                # Update running loss and accuracy
                total_loss += loss.item()
                predicted = torch.max(output.data, 1)[1]
                correct += (predicted == y_train).sum().item()

                if batch_idx % 100 == 0:
                    batch_size = len(X_train)
                    print(f'Epoch: {epoch+1} [{batch_idx*batch_size}/{len(train_loader.dataset)} ({100.*batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {100. * correct / (batch_size * (batch_idx+1)):.2f}%')
            
            # Step the scheduler
            scheduler.step()

    def test_model(self, test_loader, device='cpu'):
        self.eval()
        correct = 0

        with torch.no_grad():
            for test_imgs, test_labels in test_loader:
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
                output = self.forward(test_imgs)
                predicted = torch.max(output, 1)[1]
                correct += (predicted == test_labels).sum().item()
        
        print(f'Test accuracy: {100. * correct / len(test_loader.dataset):.2f}%')


def main():
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = os.path.join('..', '..', 'dataset')
    batch_size = 32
    train_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    print('Number of training images:', len(train_loader.dataset))
    print('Number of test images:', len(test_loader.dataset))

    model = EmotionRecognitionCNN().to(device)
    model.train_model(train_loader, lr=0.001, num_epochs=15, device=device)
    print('Train time:', time.time() - start_time)

    model.test_model(test_loader, device=device)

    torch.save(model.state_dict(), "emotion_recognition_cnn.pth")

    print('Total Execution time:', time.time() - start_time)


if __name__ == '__main__':
    main()
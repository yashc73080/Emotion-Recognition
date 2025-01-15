import os
import sys
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

sys.path.append(os.path.abspath(os.path.join('..')))

from prepare_data import prepare_data

class EmotionRecognitionCNN(nn.Module):
    def __init__(self):
        super(EmotionRecognitionCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x # maybe softmax here
    
    def train_model(self, train_loader, lr=0.001, num_epochs=5):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        error = nn.CrossEntropyLoss()
        num_epochs = num_epochs
        self.train()

        for epoch in range(num_epochs):
            correct = 0

            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                var_X_batch = Variable(X_train).float()
                var_y_batch = Variable(y_train) 
                optimizer.zero_grad()

                output = self.forward(var_X_batch)
                loss = error(output, var_y_batch)

                loss.backward()
                optimizer.step()

                # Total correct predictions
                predicted = torch.max(output.data, 1)[1] 
                correct += (predicted == var_y_batch).sum()
                if batch_idx % 50 == 0:
                    batch_size = len(X_train)
                    print(f'Epoch: {epoch+1} [{batch_idx*len(X_train)}/{len(train_loader.dataset)} ({100.*batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data:.6f}\t Accuracy:{float(correct*100) / float(batch_size*(batch_idx+1)):.3f}%')

    def test_model(self, test_loader):
        self.eval()
        correct = 0
        for test_imgs, test_labels in test_loader:
            test_imgs = Variable(test_imgs).float()
            output = self.forward(test_imgs)
            predicted = torch.max(output,1)[1]
            correct += (predicted == test_labels).sum()
        print(f'Test accuracy: {float(correct) / len(test_loader.dataset)}')

def main():
    data_dir = os.path.join('..', '..', 'dataset')
    batch_size = 32
    train_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    print('Number of training images:', len(train_loader.dataset))
    print('Number of test images:', len(test_loader.dataset))

    model = EmotionRecognitionCNN()
    model.train_model(train_loader, lr=0.001, num_epochs=5)


if __name__ == '__main__':
    main()
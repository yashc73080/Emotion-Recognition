import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
import time

sys.path.append(os.path.abspath(os.path.join('..')))

from prepare_data import prepare_data

class EmotionRecognitionCNN(nn.Module):
    def __init__(self, num_classes=7, use_batch_norm=True):
        super(EmotionRecognitionCNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Normalization
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Global Average Pooling and Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # First block
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second block with residual connection
        identity2 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third block with residual connection
        identity3 = x
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    def train_model(self, train_loader, val_loader, lr=0.001, num_epochs=5, device='cpu', weight_decay=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
        
        # Calculate class weights
        class_counts = torch.tensor([958, 111, 1024, 1774, 1233, 1247, 831])  # From your classification report
        class_weights = 1.0 / (class_counts / class_counts.sum())
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.train()

        writer = SummaryWriter()
        
        # Initialize early stopping variables
        best_val_acc = 0
        patience = 5
        counter = 0

        for epoch in range(num_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            self.train()
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                X_train, y_train = X_train.to(device), y_train.to(device)
                optimizer.zero_grad()

                output = self.forward(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()

                _, predicted = output.max(1)
                correct += predicted.eq(y_train).sum().item()
                total += y_train.size(0)
                epoch_loss += loss.item()

                if batch_idx % 100 == 0:
                    batch_size = len(X_train)
                    print(f'Epoch: {epoch+1} [{batch_idx*batch_size}/{len(train_loader.dataset)} ({100.*batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {100. * correct / (batch_size * (batch_idx+1)):.2f}%', flush=True)
            
            epoch_loss /= len(train_loader)
            epoch_accuracy = 100. * correct / total
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

            # Validation
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    output = self.forward(X_val)
                    loss = criterion(output, y_val)
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_correct += predicted.eq(y_val).sum().item()
                    val_total += y_val.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = 100. * val_correct / val_total
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            # Early stopping check
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                counter = 0
                # Save best model
                torch.save(self.state_dict(), "best_emotion_model.pth")
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            
            # Step the scheduler
            scheduler.step(val_loss)

        writer.close()

    def test_model(self, test_loader, device='cpu', class_names=None):
        self.eval()
        all_preds = []
        all_labels = []
        correct = 0
        
        with torch.no_grad():
            for test_imgs, test_labels in test_loader:
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
                output = self.forward(test_imgs)
                predicted = torch.max(output, 1)[1]
                correct += (predicted == test_labels).sum().item()
                
                # Store predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test accuracy: {accuracy:.2f}%', flush=True)
        
        # Calculate and display metrics
        if class_names is None:
            class_names = [str(i) for i in range(7)]  # Default names if not provided
            
        # Print classification report (includes precision, recall, f1-score)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        return accuracy, all_preds, all_labels


def main():
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    data_dir = os.path.join('..', '..', 'dataset')
    batch_size = 40
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    print('Number of training images:', len(train_loader.dataset), flush=True)
    print('Number of test images:', len(test_loader.dataset), flush=True)
    print('Number of validation images:', len(val_loader.dataset), flush=True)

    model = EmotionRecognitionCNN().to(device)
    model.train_model(train_loader, val_loader, lr=0.001, num_epochs=25, device=device)
    print('Train time:', time.time() - start_time, flush=True)

    model.test_model(test_loader, device=device, class_names=class_names)

    torch.save(model.state_dict(), "emotion_recognition_cnn_1.pth")

    print('Total Execution time:', time.time() - start_time)


if __name__ == '__main__':
    main()
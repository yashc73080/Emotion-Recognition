import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
import time

sys.path.append(os.path.abspath(os.path.join('..')))

from prepare_data import prepare_data

class ImprovedEmotionRecognition(nn.Module):
    def __init__(self, num_classes=7, backbone='resnet18', pretrained=True):
        super(ImprovedEmotionRecognition, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for grayscale input
        if 'resnet' in backbone:
            # For ResNet models
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            
            # Initialize new conv1 weights
            if pretrained:
                # Average RGB weights for grayscale
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        elif 'efficientnet' in backbone:
            # For EfficientNet models
            original_conv1 = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            
            if pretrained:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        original_conv1.weight.mean(dim=1, keepdim=True)
                    )
        
        # Replace final classification layer
        if 'resnet' in backbone:
            self.backbone.fc = nn.Identity()
        elif 'efficientnet' in backbone:
            self.backbone.classifier = nn.Identity()
        
        # Custom classifier with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output
    
    def train_model(self, train_loader, val_loader, lr=0.001, num_epochs=30, device='cpu', 
                   weight_decay=1e-4, warmup_epochs=3):
        
        # Use different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for pretrained features
            {'params': classifier_params, 'lr': lr}       # Higher LR for new classifier
        ], weight_decay=weight_decay)
        
        # Cosine annealing scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs-warmup_epochs, eta_min=lr*0.01
        )
        
        # Calculate class weights for imbalanced dataset
        class_counts = torch.tensor([958, 111, 1024, 1774, 1233, 1247, 831])
        class_weights = 1.0 / (class_counts / class_counts.sum())
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        
        # Use label smoothing to prevent overfitting
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        writer = SummaryWriter()
        
        # Early stopping
        best_val_acc = 0
        patience = 7
        counter = 0
        
        for epoch in range(num_epochs):
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor
            
            # Training phase
            self.train()
            epoch_loss, correct, total = 0.0, 0, 0
            
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                X_train, y_train = X_train.to(device), y_train.to(device)
                
                optimizer.zero_grad()
                output = self(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()

                _, predicted = output.max(1)
                correct += predicted.eq(y_train).sum().item()
                total += y_train.size(0)
                epoch_loss += loss.item()

                if batch_idx % 100 == 0:
                    batch_size = len(X_train)
                    current_acc = 100. * correct / total
                    print(f'Epoch: {epoch+1} [{batch_idx*batch_size}/{len(train_loader.dataset)} '
                          f'({100.*batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}\tAccuracy: {current_acc:.2f}%', flush=True)
            
            epoch_loss /= len(train_loader)
            epoch_accuracy = 100. * correct / total
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

            # Validation phase
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    output = self(X_val)
                    loss = criterion(output, y_val)
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_correct += predicted.eq(y_val).sum().item()
                    val_total += y_val.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = 100. * val_correct / val_total
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            # Learning rate scheduling (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                counter = 0
                torch.save(self.state_dict(), "best_emotion_resnet.pth")
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, '
                  f'Train Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')

        writer.close()
        return best_val_acc

    def test_model(self, test_loader, device='cpu', class_names=None):
        self.eval()
        all_preds = []
        all_labels = []
        correct = 0
        
        with torch.no_grad():
            for test_imgs, test_labels in test_loader:
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
                output = self(test_imgs)
                predicted = torch.max(output, 1)[1]
                correct += (predicted == test_labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test accuracy: {accuracy:.2f}%', flush=True)
        
        if class_names is None:
            class_names = [str(i) for i in range(7)]
            
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        return accuracy, all_preds, all_labels


def get_enhanced_transforms():
    """Enhanced data augmentation for better generalization"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform


def main():
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}', flush=True)

    data_dir = os.path.join('..', '..', 'dataset')
    batch_size = 32  # Slightly smaller batch size for stability
    
    # You might need to modify prepare_data to accept transforms
    train_loader, val_loader, test_loader = prepare_data(data_dir, batch_size=batch_size)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    print('Number of training images:', len(train_loader.dataset), flush=True)
    print('Number of test images:', len(test_loader.dataset), flush=True)
    print('Number of validation images:', len(val_loader.dataset), flush=True)

    # Try different backbones - ResNet18 is good balance of speed and accuracy
    model = ImprovedEmotionRecognition(
        num_classes=7, 
        backbone='resnet50',  # Try 'resnet34', 'resnet50', or 'efficientnet_b0'
        pretrained=True
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    best_val_acc = model.train_model(
        train_loader, val_loader, 
        lr=0.001, 
        num_epochs=30, 
        device=device,
        weight_decay=1e-4
    )
    
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    print('Train time:', time.time() - start_time, flush=True)

    # Load best model for testing
    model.load_state_dict(torch.load("best_emotion_resnet.pth"))
    test_acc, _, _ = model.test_model(test_loader, device=device, class_names=class_names)
    
    print(f'Final test accuracy: {test_acc:.2f}%')
    print('Total Execution time:', time.time() - start_time)


if __name__ == '__main__':
    main()
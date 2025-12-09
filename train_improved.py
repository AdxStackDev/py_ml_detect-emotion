import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import os
from datetime import datetime

# ================================
# 1. Device Configuration
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ================================
# 2. Dataset & Dataloader
# ================================
data_dir = "emotion_dataset"

# Enhanced transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Detected classes: {class_names}")

# Calculate class distribution
class_counts = {}
for _, label in full_dataset.samples:
    class_name = class_names[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

print("\nClass Distribution:")
total_samples = len(full_dataset)
for class_name in class_names:
    count = class_counts[class_name]
    percentage = (count / total_samples) * 100
    print(f"  {class_name}: {count} images ({percentage:.1f}%)")

# Calculate class weights for balanced loss
class_weights = []
for i in range(num_classes):
    class_name = class_names[i]
    weight = total_samples / (num_classes * class_counts[class_name])
    class_weights.append(weight)
    
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"\nClass weights for balanced loss: {class_weights.cpu().numpy()}")

# Split dataset into train and validation (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply validation transform to validation set
val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

print(f"\nDataset split:")
print(f"  Training samples: {train_size}")
print(f"  Validation samples: {val_size}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ================================
# 3. Improved CNN Model with Dropout and Batch Normalization
# ================================
class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ImprovedEmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [batch, 32, 24, 24]
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [batch, 64, 12, 12]
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [batch, 128, 6, 6]
        
        # Flatten
        x = x.view(-1, 128 * 6 * 6)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ================================
# 4. Training and Validation Functions
# ================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ================================
# 5. Training Setup
# ================================
model = ImprovedEmotionCNN(num_classes=num_classes).to(device)

# Use class-weighted loss to handle imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# ================================
# 6. Training Loop with Early Stopping
# ================================
EPOCHS = 50
best_val_loss = float('inf')
best_val_acc = 0.0
patience = 10
patience_counter = 0

# History tracking
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

print(f"\n{'='*60}")
print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
    
    # Early stopping and model checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'class_names': class_names
        }, 'best_emotion_model.pth')
        print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"  Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print()

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"{'='*60}\n")

# ================================
# 7. Load Best Model and Final Evaluation
# ================================
checkpoint = torch.load('best_emotion_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

# Final validation
val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)

# ================================
# 8. Generate Evaluation Metrics
# ================================
print("\n" + "="*60)
print("FINAL EVALUATION METRICS")
print("="*60 + "\n")

# Classification report
print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(val_labels, val_preds)
print("\nConfusion Matrix:")
print(cm)

# ================================
# 9. Visualization
# ================================
# Create output directory
os.makedirs('training_results', exist_ok=True)

# Plot 1: Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results/training_history.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved training history plot to 'training_results/training_history.png'")

# Plot 2: Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix to 'training_results/confusion_matrix.png'")

# Plot 3: Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(8, 5))
bars = plt.bar(class_names, per_class_acc, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
plt.xlabel('Emotion Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('training_results/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved per-class accuracy to 'training_results/per_class_accuracy.png'")

# ================================
# 10. Save Final Model for Inference
# ================================
# Save just the state dict for easy loading
torch.save(model.state_dict(), 'emotion_model.pth')
print("\n✓ Final model saved as 'emotion_model.pth'")

# Save model info
with open('training_results/model_info.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("EMOTION DETECTION MODEL - TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Total Epochs Trained: {len(history['train_loss'])}\n")
    f.write(f"Best Epoch: {checkpoint['epoch']+1}\n\n")
    f.write(f"Classes: {class_names}\n")
    f.write(f"Number of Classes: {num_classes}\n\n")
    f.write(f"Dataset Split:\n")
    f.write(f"  Training samples: {train_size}\n")
    f.write(f"  Validation samples: {val_size}\n\n")
    f.write(f"Class Distribution:\n")
    for class_name in class_names:
        count = class_counts[class_name]
        percentage = (count / total_samples) * 100
        f.write(f"  {class_name}: {count} images ({percentage:.1f}%)\n")
    f.write(f"\nFinal Metrics:\n")
    f.write(f"  Best Validation Loss: {best_val_loss:.4f}\n")
    f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n\n")
    f.write(f"Per-Class Accuracy:\n")
    for i, class_name in enumerate(class_names):
        f.write(f"  {class_name}: {per_class_acc[i]:.2%}\n")

print("✓ Saved model info to 'training_results/model_info.txt'")

print("\n" + "="*60)
print("ALL DONE! 🎉")
print("="*60)

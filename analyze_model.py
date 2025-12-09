import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# ============================
# Model Definition
# ============================
class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(ImprovedEmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ============================
# Analysis Functions
# ============================
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size(model_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def analyze_dataset(data_dir):
    """Analyze dataset distribution"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    
    # Count samples per class
    class_counts = {}
    for _, label in dataset.samples:
        class_name = class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_names, class_counts, len(dataset)

def visualize_sample_predictions(model, data_dir, class_names, device, num_samples=9):
    """Visualize sample predictions"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    
    # Get one batch
    images, labels = next(iter(loader))
    images = images.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
    
    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().squeeze().numpy()
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i][preds[i]].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})"
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ============================
# Main Analysis
# ============================
def main():
    print("="*60)
    print("EMOTION DETECTION MODEL ANALYSIS")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 1. Dataset Analysis
    print("1. DATASET ANALYSIS")
    print("-" * 60)
    data_dir = "emotion_dataset"
    
    if os.path.exists(data_dir):
        class_names, class_counts, total_samples = analyze_dataset(data_dir)
        
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}\n")
        
        print("Class Distribution:")
        for class_name in class_names:
            count = class_counts[class_name]
            percentage = (count / total_samples) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {class_name:8s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print("⚠ Warning: Significant class imbalance detected!")
            print("  Recommendation: Use class-weighted loss or resampling")
    else:
        print(f"Dataset directory '{data_dir}' not found!")
        class_names = ['angry', 'happy', 'sad']
    
    print()
    
    # 2. Model Architecture Analysis
    print("2. MODEL ARCHITECTURE ANALYSIS")
    print("-" * 60)
    
    model = ImprovedEmotionCNN(num_classes=len(class_names))
    total_params, trainable_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture:")
    print(model)
    print()
    
    # 3. Model File Analysis
    print("3. TRAINED MODEL ANALYSIS")
    print("-" * 60)
    
    model_path = "emotion_model.pth"
    if os.path.exists(model_path):
        model_size = get_model_size(model_path)
        print(f"Model file: {model_path}")
        print(f"Model size: {model_size:.2f} MB")
        
        # Load model
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("✓ Model loaded successfully!")
            
            # Test inference speed
            dummy_input = torch.randn(1, 1, 48, 48).to(device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Measure
            import time
            num_runs = 100
            start = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input)
            end = time.time()
            
            avg_time = (end - start) / num_runs * 1000  # ms
            fps = 1000 / avg_time
            
            print(f"\nInference Speed:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    else:
        print(f"Model file '{model_path}' not found!")
        print("Please train the model first using 'train_improved.py'")
    
    print()
    
    # 4. Visualize Sample Predictions
    if os.path.exists(model_path) and os.path.exists(data_dir):
        print("4. SAMPLE PREDICTIONS VISUALIZATION")
        print("-" * 60)
        print("Generating sample predictions...")
        
        os.makedirs('analysis_results', exist_ok=True)
        
        fig = visualize_sample_predictions(model, data_dir, class_names, device)
        fig.savefig('analysis_results/sample_predictions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved to 'analysis_results/sample_predictions.png'")
        plt.close()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

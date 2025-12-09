"""
Simple ONNX Export Script for Emotion Detection Model
Exports PyTorch model to ONNX format with minimal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Set environment variable to avoid Unicode issues
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ============================
# Model Definition
# ============================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# Export Function
# ============================
def export_model():
    print("="*60)
    print("PYTORCH TO ONNX EXPORT")
    print("="*60)
    
    # Load PyTorch model
    print("\n[1/4] Loading PyTorch model...")
    device = torch.device("cpu")
    model = EmotionCNN(num_classes=3)
    model.load_state_dict(torch.load('sad_happy_angry.pth', map_location=device))
    model.eval()
    print("  OK - Model loaded")
    
    # Create dummy input
    print("\n[2/4] Creating dummy input...")
    dummy_input = torch.randn(1, 1, 48, 48)
    print("  OK - Input shape: (1, 1, 48, 48)")
    
    # Export to ONNX
    print("\n[3/4] Exporting to ONNX...")
    output_path = 'emotion_model.onnx'
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )
        print(f"  OK - Exported to {output_path}")
    except Exception as e:
        print(f"  ERROR - Export failed: {e}")
        return False
    
    # Verify file exists
    print("\n[4/4] Verifying export...")
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  OK - File created: {file_size:.2f} MB")
    else:
        print("  ERROR - File not created")
        return False
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nONNX model saved as: {output_path}")
    print("\nNext steps:")
    print("  1. Test with: python test_onnx.py")
    print("  2. Deploy to your platform")
    
    return True

if __name__ == "__main__":
    export_model()

"""
Export PyTorch Emotion Detection Model to ONNX Format

This script converts the trained PyTorch model to ONNX format for:
- Faster inference
- Cross-platform deployment
- Integration with other frameworks (TensorFlow.js, ONNX Runtime, etc.)
- Mobile and edge device deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# ============================
# 1. Define Model Architecture
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
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 12, 12]
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# 2. Load Trained Model
# ============================
def load_pytorch_model(model_path='sad_happy_angry.pth', num_classes=3):
    """Load the trained PyTorch model"""
    device = torch.device("cpu")  # Use CPU for export
    model = EmotionCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[OK] PyTorch model loaded from {model_path}")
    return model

# ============================
# 3. Export to ONNX
# ============================
def export_to_onnx(model, onnx_path='emotion_model.onnx', input_shape=(1, 1, 48, 48)):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        onnx_path: Output path for ONNX model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Dummy input
        onnx_path,                      # Output file path
        export_params=True,             # Store trained parameters
        opset_version=12,               # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input tensor name
        output_names=['output'],        # Output tensor name
        dynamic_axes={                  # Dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"[OK] Model exported to ONNX: {onnx_path}")
    return onnx_path

# ============================
# 4. Verify ONNX Model
# ============================
def verify_onnx_model(onnx_path='emotion_model.onnx'):
    """Verify the exported ONNX model"""
    # Load and check the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model verified successfully")
    
    # Print model info
    print("\nModel Information:")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Opset Version: {onnx_model.opset_import[0].version}")
    
    # Print input/output info
    print("\nInput:")
    for input_tensor in onnx_model.graph.input:
        print(f"  Name: {input_tensor.name}")
        print(f"  Shape: {[dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]}")
    
    print("\nOutput:")
    for output_tensor in onnx_model.graph.output:
        print(f"  Name: {output_tensor.name}")
        print(f"  Shape: {[dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]}")

# ============================
# 5. Test ONNX Model
# ============================
def test_onnx_inference(onnx_path='emotion_model.onnx', test_image_path='boy.png'):
    """Test inference with ONNX Runtime"""
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Prepare test image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(test_image_path).convert("L")
    img_tensor = transform(image).unsqueeze(0).numpy()
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Get predictions
    output = ort_outputs[0]
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)  # Softmax
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    class_names = ['angry', 'happy', 'sad']
    
    print(f"\n[OK] ONNX Inference Test:")
    print(f"  Image: {test_image_path}")
    print(f"  Predicted Emotion: {class_names[predicted_class]}")
    print(f"  Confidence: {probabilities[0][predicted_class]:.2%}")
    print(f"\n  Probability Distribution:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name:8s}: {probabilities[0][i]:.2%}")

# ============================
# 6. Compare PyTorch vs ONNX
# ============================
def compare_models(pytorch_model, onnx_path='emotion_model.onnx', test_image_path='boy.png'):
    """Compare PyTorch and ONNX model outputs"""
    # Prepare test image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(test_image_path).convert("L")
    img_tensor = transform(image).unsqueeze(0)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(img_tensor).numpy()
    
    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    difference = np.abs(pytorch_output - onnx_output).max()
    
    print(f"\n[OK] Model Comparison:")
    print(f"  PyTorch Output: {pytorch_output[0]}")
    print(f"  ONNX Output:    {onnx_output[0]}")
    print(f"  Max Difference: {difference:.6f}")
    
    if difference < 1e-5:
        print(f"  Status: ✓ Models match perfectly!")
    elif difference < 1e-3:
        print(f"  Status: ✓ Models match (minor numerical differences)")
    else:
        print(f"  Status: ⚠ Models have significant differences")

# ============================
# 7. Get Model Size
# ============================
def get_model_sizes(pytorch_path='sad_happy_angry.pth', onnx_path='emotion_model.onnx'):
    """Compare model file sizes"""
    import os
    
    pytorch_size = os.path.getsize(pytorch_path) / (1024 * 1024)  # MB
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    
    print(f"\n[OK] Model Sizes:")
    print(f"  PyTorch (.pth): {pytorch_size:.2f} MB")
    print(f"  ONNX (.onnx):   {onnx_size:.2f} MB")
    print(f"  Difference:     {abs(pytorch_size - onnx_size):.2f} MB")

# ============================
# 8. Main Export Function
# ============================
def main():
    """Main function to export and verify ONNX model"""
    print("="*60)
    print("PYTORCH TO ONNX MODEL EXPORT")
    print("="*60 + "\n")
    
    # Step 1: Load PyTorch model
    print("Step 1: Loading PyTorch model...")
    pytorch_model = load_pytorch_model('sad_happy_angry.pth', num_classes=3)
    
    # Step 2: Export to ONNX
    print("\nStep 2: Exporting to ONNX...")
    onnx_path = export_to_onnx(pytorch_model, 'emotion_model.onnx')
    
    # Step 3: Verify ONNX model
    print("\nStep 3: Verifying ONNX model...")
    verify_onnx_model(onnx_path)
    
    # Step 4: Test ONNX inference
    print("\nStep 4: Testing ONNX inference...")
    test_onnx_inference(onnx_path, 'boy.png')
    
    # Step 5: Compare models
    print("\nStep 5: Comparing PyTorch vs ONNX...")
    compare_models(pytorch_model, onnx_path, 'boy.png')
    
    # Step 6: Compare file sizes
    print("\nStep 6: Comparing file sizes...")
    get_model_sizes('sad_happy_angry.pth', onnx_path)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nYour ONNX model is ready: {onnx_path}")
    print("\nNext steps:")
    print("  1. Use ONNX Runtime for faster inference")
    print("  2. Deploy to web with ONNX.js")
    print("  3. Deploy to mobile with ONNX Runtime Mobile")
    print("  4. Convert to TensorFlow.js if needed")

if __name__ == "__main__":
    main()

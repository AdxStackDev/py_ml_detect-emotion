import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

# ============================
# 1. Device Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. Define ImprovedEmotionCNN Model
# ============================
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 6 * 6)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ============================
# 3. Preprocessing Transformation
# ============================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================
# 4. Inference Function
# ============================
def predict_emotion(image_path, model, class_names, device, show_probabilities=False):
    """
    Predict emotion from an image file.
    
    Args:
        image_path: Path to the image file
        model: Trained PyTorch model
        class_names: List of emotion class names
        device: Device to run inference on
        show_probabilities: Whether to show probability distribution
    
    Returns:
        predicted_class: Predicted emotion label
        probabilities: Probability distribution (if show_probabilities=True)
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(output, 1).item()
            predicted_class = class_names[pred_idx]
        
        if show_probabilities:
            return predicted_class, probabilities
        else:
            return predicted_class
            
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        return None

# ============================
# 5. Main Execution
# ============================
def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Inference')
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--images', nargs='+', help='Paths to multiple image files')
    parser.add_argument('--model', type=str, default='emotion_model.pth', 
                        help='Path to model weights (default: emotion_model.pth)')
    parser.add_argument('--classes', nargs='+', default=['angry', 'happy', 'sad'],
                        help='Class names (default: angry happy sad)')
    parser.add_argument('--show-probs', action='store_true',
                        help='Show probability distribution for each prediction')
    
    args = parser.parse_args()
    
    # Determine which images to process
    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = args.images
    else:
        # Default test images
        image_paths = ['crying.png', 'boy.png', 'person.png']
        print("No images specified. Using default test images.")
    
    # Load model
    print(f"Loading model from '{args.model}'...")
    try:
        class_names = args.classes
        num_classes = len(class_names)
        
        model = ImprovedEmotionCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully!")
        print(f"  Device: {device}")
        print(f"  Classes: {class_names}\n")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        print("Please train the model first using 'train_improved.py'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Process images
    print("="*60)
    print("EMOTION DETECTION RESULTS")
    print("="*60 + "\n")
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"⚠ Image '{img_path}' not found. Skipping...")
            continue
        
        if args.show_probs:
            result = predict_emotion(img_path, model, class_names, device, show_probabilities=True)
            if result:
                predicted_class, probabilities = result
                print(f"Image: {img_path}")
                print(f"  Predicted Emotion: {predicted_class.upper()}")
                print(f"  Probability Distribution:")
                for i, class_name in enumerate(class_names):
                    bar = "█" * int(probabilities[i] * 50)
                    print(f"    {class_name:8s}: {probabilities[i]:.2%} {bar}")
                print()
        else:
            predicted_class = predict_emotion(img_path, model, class_names, device)
            if predicted_class:
                print(f"Image: '{img_path}' => Predicted Emotion: {predicted_class.upper()}")
    
    print("="*60)

if __name__ == "__main__":
    main()

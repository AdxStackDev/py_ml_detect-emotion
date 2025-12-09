from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import uuid
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['RESULTS_FILE'] = 'results.json'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 12, 12]
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# Load Model
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['angry', 'happy', 'sad']
model = EmotionCNN(num_classes=len(class_names)).to(device)

try:
    model.load_state_dict(torch.load('sad_happy_angry.pth', map_location=device))
    model.eval()
    print(f"[OK] Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")

# ============================
# Image Preprocessing
# ============================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================
# Helper Functions
# ============================
def get_image_metadata(image_path, original_filename):
    """Extract image metadata"""
    try:
        img = Image.open(image_path)
        file_size = os.path.getsize(image_path)
        
        return {
            'filename': original_filename,
            'size': f"{file_size / 1024:.2f} KB",
            'size_bytes': file_size,
            'resolution': f"{img.width} x {img.height}",
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode
        }
    except Exception as e:
        return {'error': str(e)}

def predict_emotion(image_path):
    """Predict emotion from image"""
    try:
        image = Image.open(image_path).convert("L")
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(output, 1).item()
            predicted_class = class_names[pred_idx]
            confidence = probabilities[pred_idx]
        
        # Create probability distribution
        prob_dist = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return {
            'emotion': predicted_class,
            'confidence': float(confidence),
            'probabilities': prob_dist,
            'all_emotions': class_names
        }
    except Exception as e:
        return {'error': str(e)}

def save_result(result_data):
    """Save result to JSON file"""
    try:
        if os.path.exists(app.config['RESULTS_FILE']):
            with open(app.config['RESULTS_FILE'], 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        results.append(result_data)
        
        with open(app.config['RESULTS_FILE'], 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving result: {e}")

def get_result_by_id(result_id):
    """Get a specific result by ID"""
    try:
        if os.path.exists(app.config['RESULTS_FILE']):
            with open(app.config['RESULTS_FILE'], 'r') as f:
                results = json.load(f)
            
            for result in results:
                if result['id'] == result_id:
                    return result
        return None
    except Exception as e:
        print(f"Error getting result: {e}")
        return None

# ============================
# Routes
# ============================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/details/<result_id>')
def details(result_id):
    """Details page for a specific result"""
    return render_template('details.html', result_id=result_id)

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """Handle image upload and processing"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400
    
    results = []
    
    for file in files:
        if file and file.filename:
            # Generate unique ID and filename
            result_id = str(uuid.uuid4())
            ext = os.path.splitext(file.filename)[1]
            unique_filename = f"{result_id}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save file
            file.save(filepath)
            
            # Get metadata
            metadata = get_image_metadata(filepath, file.filename)
            
            # Predict emotion
            prediction = predict_emotion(filepath)
            
            # Create result object
            result = {
                'id': result_id,
                'filename': unique_filename,
                'original_filename': file.filename,
                'upload_time': datetime.now().isoformat(),
                'metadata': metadata,
                'prediction': prediction
            }
            
            # Save to persistent storage
            save_result(result)
            
            results.append(result)
    
    return jsonify({
        'success': True,
        'count': len(results),
        'results': results
    })

@app.route('/api/result/<result_id>')
def get_result(result_id):
    """Get a specific result by ID"""
    result = get_result_by_id(result_id)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Result not found'}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/history')
def get_history():
    """Get all processing history"""
    try:
        if os.path.exists(app.config['RESULTS_FILE']):
            with open(app.config['RESULTS_FILE'], 'r') as f:
                results = json.load(f)
            return jsonify({'results': results})
        else:
            return jsonify({'results': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

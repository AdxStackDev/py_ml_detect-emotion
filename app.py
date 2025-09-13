import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from flask import Flask, request, render_template, jsonify
import os

# ============================
# 1. Flask App Initialization
# ============================
app = Flask(__name__)

# ============================
# 2. Device Configuration
# ============================
device = torch.device("cpu")

# ============================
# 3. Label Classes
# ============================
class_names = ['angry', 'happy', 'sad']

# ============================
# 4. Define EmotionCNN Model
# ============================
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, len(class_names))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================
# 5. Preprocessing Transformation
# ============================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================
# 6. Load Model
# ============================
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("sad_happy_angry.pth", map_location=device))
model.eval()

# ============================
# 7. Prediction Function
# ============================
def predict_emotion(image_path):
    try:
        image = Image.open(image_path).convert("L")
        # The original script inverted the image. Let's see if this is needed.
        # It was mentioned that it's for white backgrounds. Let's assume it's needed for now.
        # image = ImageOps.invert(image)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, 1).item()
            return class_names[pred]
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return None

# ============================
# 8. Flask Routes
# ============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        prediction = predict_emotion(filepath)

        if prediction:
            return jsonify({'emotion': prediction})
        else:
            return jsonify({'error': 'Could not predict emotion'})

if __name__ == '__main__':
    app.run(debug=True)

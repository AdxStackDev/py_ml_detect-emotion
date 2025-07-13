import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps

# ============================
# 1. Device Configuration
# ============================
device = torch.device("cpu")
# print("Using device:", device)

# ============================
# 2. Label Classes (Make sure they match your training set)
# ============================
class_names = ['angry', 'happy', 'sad']

# ============================
# 3. Define EmotionCNN Model
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
# 4. Preprocessing Transformation
# ============================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ============================
# 5. Load Model
# ============================
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("sad_happy_angry.pth", map_location=device))
model.eval()

# ============================
# 6. Predict on Custom Image
# ============================
img_path = ['crying.png', 'boy.png', 'person.png']

for imgemotion in img_path:

    image = Image.open(imgemotion).convert("L")
    image = ImageOps.invert(image)  # If your image has white background

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, 1).item()
        print(f"image:'{imgemotion}' => Predicted emotion: {class_names[pred]}")

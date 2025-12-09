import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ================================
# 1. Device Configuration
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================================
# 2. Dataset & Dataloader
# ================================
data_dir = "emotion_dataset"  # Folder with subfolders: happy/, sad/, angry/, crying/

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert RGB to grayscale
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes  # ['angry', 'crying', 'happy', 'sad']
print("Detected classes:", class_names)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ================================
# 3. CNN Model
# ================================
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, len(class_names))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 12, 12]
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ================================
# 4. Train the Model
# ================================
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "sad_happy_angry.pth")
print("Model saved as sad_happy_angry.pth")

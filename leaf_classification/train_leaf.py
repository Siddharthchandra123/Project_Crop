import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from leaf_model import LeafClassifier

# Config
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.0001

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
leaf_dataset = r"C:\Users\Lenovo\.cache\kagglehub\datasets\rashidthihan\plant-disease-dataset\versions\1\Plant_Disease_Dataset\train"
dataset = datasets.ImageFolder(leaf_dataset, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeafClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} - Accuracy: {acc:.2f}%")

# Save model
torch.save(model.state_dict(), "leaf_classifier.pth")
print("✅ Leaf classifier saved")

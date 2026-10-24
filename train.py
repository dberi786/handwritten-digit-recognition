
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from model import SimpleCNN


# Device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)


# Load dataset
transform = transforms.ToTensor()

train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


# Create model
model = SimpleCNN().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 3
history = {"loss": []}

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    history["loss"].append(average_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Loss: {average_loss:.4f}"
    )


# Save model and training history
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/simple_cnn_mnist.pth"
)

with open("results/training_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("Model saved successfully!")
print("Training history saved!")

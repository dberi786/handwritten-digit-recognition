
import os
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model import SimpleCNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load test data
test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

# Load trained model
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load(
        "models/simple_cnn_mnist.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

all_predictions = []
all_labels = []
wrong_images = []
wrong_predictions = []
wrong_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

        # Store some incorrectly classified images
        wrong_indices = (
            predictions != labels
        ).nonzero(as_tuple=True)[0]

        for idx in wrong_indices:
            if len(wrong_images) < 10:
                wrong_images.append(images[idx].cpu().squeeze())
                wrong_predictions.append(
                    predictions[idx].item()
                )
                wrong_labels.append(labels[idx].item())


# Metrics
accuracy = accuracy_score(
    all_labels, all_predictions
)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4,
        zero_division=0
    )
)


# Confusion matrix
os.makedirs("results", exist_ok=True)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(10))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
)

fig, ax = plt.subplots(figsize=(9, 9))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("MNIST CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "results/confusion_matrix.png",
    dpi=150
)
plt.show()


# Display incorrect predictions
if wrong_images:
    fig, axes = plt.subplots(
        2, 5, figsize=(12, 6)
    )

    for i, ax in enumerate(axes.flat):
        if i < len(wrong_images):
            ax.imshow(
                wrong_images[i],
                cmap="gray"
            )
            ax.set_title(
                f"Pred: {wrong_predictions[i]} | "
                f"Actual: {wrong_labels[i]}",
                color="red"
            )
        ax.axis("off")

    plt.suptitle("Incorrect Predictions")
    plt.tight_layout()
    plt.savefig(
        "results/incorrect_predictions.png",
        dpi=150
    )
    plt.show()

import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

TRAIN_GOOD_PATH = "dataset/bottle/train/good"
TEST_PATH = "dataset/bottle/test"

MODEL_SAVE_PATH = "models/defect_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {DEVICE}")

# ------------------------------------------------
# IMAGE TRANSFORMS
# ------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------
# LOAD IMAGE PATHS + LABELS
# ------------------------------------------------

image_paths = []
labels = []

# ------------------------------------------------
# GOOD IMAGES
# ------------------------------------------------

for file in os.listdir(TRAIN_GOOD_PATH):

    image_paths.append(
        os.path.join(TRAIN_GOOD_PATH, file)
    )

    labels.append(0)

# ------------------------------------------------
# TEST IMAGES
# ------------------------------------------------

for defect_type in os.listdir(TEST_PATH):

    defect_folder = os.path.join(
        TEST_PATH,
        defect_type
    )

    if os.path.isdir(defect_folder):

        for file in os.listdir(defect_folder):

            image_paths.append(
                os.path.join(defect_folder, file)
            )

            if defect_type == "good":

                labels.append(0)

            else:

                labels.append(1)

print(f"Total Images: {len(image_paths)}")

# ------------------------------------------------
# TRAIN VALIDATION SPLIT
# ------------------------------------------------

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ------------------------------------------------
# CUSTOM DATASET
# ------------------------------------------------

class BottleDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label

# ------------------------------------------------
# DATASETS
# ------------------------------------------------

train_dataset = BottleDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = BottleDataset(
    val_paths,
    val_labels,
    val_transform
)

# ------------------------------------------------
# DATALOADERS
# ------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = models.resnet18(
    weights="DEFAULT"
)

# MODIFY FINAL LAYER

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)

# ------------------------------------------------
# LOSS + OPTIMIZER
# ------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------

best_val_accuracy = 0

print("Starting Training...\n")

for epoch in range(EPOCHS):

    # ------------------------------------------------
    # TRAINING
    # ------------------------------------------------

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    # ------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("Best Model Saved!")

    # ------------------------------------------------
    # PRINT METRICS
    # ------------------------------------------------

    print(f"Epoch [{epoch+1}/{EPOCHS}]")

    print(
        f"Training Loss: "
        f"{running_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print("-" * 40)

print("\nTraining Completed!")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)
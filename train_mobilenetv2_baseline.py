
import os
import copy
import random
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(r"D:\VISIONX")

DATA_DIR = ROOT / "dataset_split"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["angry", "fear", "happy", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 8

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_5class_baseline_best.pth"
)

# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# 3. TRANSFORMS
# ==========================================

# Baseline: no CLAHE, alignment or occlusion.
# Only standard resizing, flipping and normalization.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

# Check class names and order
assert train_dataset.classes == CLASSES, (
    f"Unexpected classes: {train_dataset.classes}"
)

assert val_dataset.classes == CLASSES
assert test_dataset.classes == CLASSES

print("\nClasses:", train_dataset.classes)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

targets = train_dataset.targets

class_counts = np.bincount(
    targets,
    minlength=NUM_CLASSES
)

class_weights = 1.0 / np.maximum(class_counts, 1)

sample_weights = [
    class_weights[label] for label in targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ==========================================
# 6. DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# ==========================================
# 7. MOBILENETV2 BASELINE MODEL
# ==========================================

# weights=None means random initialization.
# No pretrained ImageNet weights.
# No CBAM or additional attention modules.

model = models.mobilenet_v2(weights=None)

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

print("\nMobileNetV2 Baseline Model Created")
print("Training from scratch: True")
print("CBAM: False")

# ==========================================
# 8. LOSS, OPTIMIZER, SCHEDULER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)

# ==========================================
# 9. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ==========================================
# 10. VALIDATION FUNCTION
# ==========================================

def validate(model, loader):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_f1

# ==========================================
# 11. TRAINING LOOP
# ==========================================

best_val_f1 = -1.0
best_epoch = 0
patience_counter = 0

history = []

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc * 100:.2f}% | "
        f"Val Macro F1: {val_f1:.4f}"
    )

    print(f"Learning Rate: {current_lr:.7f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    # Save best model using validation macro F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASSES,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "training_from_scratch": True,
            "model_name": "MobileNetV2 Baseline"
        }, BEST_MODEL_PATH)

        print("Best baseline model saved!")

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break

# ==========================================
# 12. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULT_DIR / "mobilenetv2_baseline_history.csv",
    index=False
)

# ==========================================
# 13. LOAD BEST MODEL
# ==========================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nBest epoch:", checkpoint["best_epoch"])

# ==========================================
# 14. FINAL TEST EVALUATION
# ==========================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# ==========================================
# 15. CALCULATE METRICS
# ==========================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASSES,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n========== BASELINE RESULTS ==========")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 16. SAVE RESULTS
# ==========================================

with open(
    RESULT_DIR / "mobilenetv2_baseline_report.txt",
    "w"
) as file:

    file.write("MobileNetV2 Baseline Results\n\n")
    file.write(f"Best Epoch: {best_epoch}\n")
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    file.write(f"Test Macro F1: {test_f1:.4f}\n\n")
    file.write(report)

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(
    RESULT_DIR / "mobilenetv2_baseline_confusion_matrix.csv"
)

print("\nBaseline training completed!")
print("Model:", BEST_MODEL_PATH)
print("Results:", RESULT_DIR)

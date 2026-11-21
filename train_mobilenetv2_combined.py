
import random
import numpy as np
import pandas as pd
import cv2

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
NUM_CLASSES = 5

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 8

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
SEED = 42

BEST_MODEL_PATH = MODEL_DIR / "mobilenetv2_5class_combined_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 3. FACIAL ALIGNMENT
# ==========================================

# IMPORTANT: Load XML files from your project folder.
CASCADE_DIR = MODEL_DIR / "haarcascades"

FACE_CASCADE_PATH = CASCADE_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = CASCADE_DIR / "haarcascade_eye.xml"

if not FACE_CASCADE_PATH.is_file():
    raise FileNotFoundError(
        f"Face cascade not found: {FACE_CASCADE_PATH}"
    )

if not EYE_CASCADE_PATH.is_file():
    raise FileNotFoundError(
        f"Eye cascade not found: {EYE_CASCADE_PATH}"
    )

face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))

if face_cascade.empty():
    raise RuntimeError("Face cascade failed to load!")

if eye_cascade.empty():
    raise RuntimeError("Eye cascade failed to load!")

print("Haar Cascade files loaded successfully!")


def align_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image

    # Select largest face
    x, y, w, h = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    face_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(8, 8)
    )

    if len(eyes) < 2:
        return image

    # Select two largest eye detections
    eyes = sorted(
        eyes,
        key=lambda box: box[2] * box[3],
        reverse=True
    )[:2]

    eyes = sorted(eyes, key=lambda box: box[0])

    left_eye = (
        x + eyes[0][0] + eyes[0][2] / 2,
        y + eyes[0][1] + eyes[0][3] / 2
    )

    right_eye = (
        x + eyes[1][0] + eyes[1][2] / 2,
        y + eyes[1][1] + eyes[1][3] / 2
    )

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    aligned = cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return aligned


# ==========================================
# 4. CLAHE
# ==========================================

def apply_clahe(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    result = cv2.merge([enhanced_l, a, b])

    result = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB
    )

    return result


# ==========================================
# 5. COMBINED PREPROCESSING
# ==========================================

class CombinedPreprocessing:

    def __call__(self, pil_image):

        image = np.array(pil_image.convert("RGB"))

        # Facial alignment
        image = align_face(image)

        # CLAHE enhancement
        image = apply_clahe(image)

        return Image.fromarray(image)


# ==========================================
# 6. CBAM CHANNEL ATTENTION
# ==========================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


# ==========================================
# 7. CBAM SPATIAL ATTENTION
# ==========================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out = torch.max(
            x,
            dim=1,
            keepdim=True
        )[0]

        combined = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.sigmoid(self.conv(combined))

        return x * attention


# ==========================================
# 8. COMPLETE CBAM
# ==========================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


# ==========================================
# 9. MOBILENETV2 + CBAM
# ==========================================

class MobileNetV2CBAM(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        # Train from scratch: no pretrained weights
        self.backbone = models.mobilenet_v2(weights=None)

        channels = self.backbone.last_channel

        self.cbam = CBAM(channels)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            channels,
            num_classes
        )

    def forward(self, x):

        x = self.backbone.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


# ==========================================
# 10. DATA TRANSFORMS
# ==========================================

train_transform = transforms.Compose([

    CombinedPreprocessing(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    # Occlusion augmentation
    transforms.RandomErasing(
        p=0.35,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    )
])


# Validation and test: no random occlusion
eval_transform = transforms.Compose([

    CombinedPreprocessing(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 11. LOAD DATASET
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

assert train_dataset.classes == CLASSES, train_dataset.classes
assert val_dataset.classes == CLASSES, val_dataset.classes
assert test_dataset.classes == CLASSES, test_dataset.classes

print("\nClasses:", CLASSES)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))


# ==========================================
# 12. CLASS IMBALANCE HANDLING
# ==========================================

targets = train_dataset.targets

class_counts = np.bincount(
    targets,
    minlength=NUM_CLASSES
)

class_weights = 1.0 / np.maximum(class_counts, 1)

sample_weights = [
    class_weights[label]
    for label in targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ==========================================
# 13. DATA LOADERS
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
# 14. CREATE MODEL
# ==========================================

model = MobileNetV2CBAM(
    num_classes=NUM_CLASSES
).to(DEVICE)

print("\nCombined Model Created")
print("MobileNetV2: Enabled")
print("CLAHE: Enabled")
print("Facial Alignment: Enabled")
print("Occlusion: Enabled")
print("CBAM: Enabled")
print("Training from scratch: True")


# ==========================================
# 15. LOSS AND OPTIMIZER
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
# 16. TRAINING FUNCTION
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

    return total_loss / total, correct / total


# ==========================================
# 17. VALIDATION FUNCTION
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

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):

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
# 18. TRAINING LOOP
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
            "model_name": "MobileNetV2 Combined",
            "clahe": True,
            "face_alignment": True,
            "occlusion_augmentation": True,
            "cbam": True

        }, BEST_MODEL_PATH)

        print("Best combined model saved!")

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==========================================
# 19. SAVE TRAINING HISTORY
# ==========================================

pd.DataFrame(history).to_csv(
    RESULT_DIR / "mobilenetv2_combined_history.csv",
    index=False
)


# ==========================================
# 20. LOAD BEST MODEL
# ==========================================

if best_epoch == 0:
    raise RuntimeError("Training did not save a best model.")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nBest Epoch:", checkpoint["best_epoch"])


# ==========================================
# 21. FINAL TEST EVALUATION
# ==========================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ==========================================
# 22. METRICS
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
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\n========== COMBINED RESULTS ==========")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 23. SAVE RESULTS
# ==========================================

with open(
    RESULT_DIR / "mobilenetv2_combined_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MobileNetV2 + CLAHE + Alignment + Occlusion + CBAM\n\n"
    )

    file.write(f"Best Epoch: {best_epoch}\n")

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Test Macro F1: {test_f1:.4f}\n\n"
    )

    file.write(report)


pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(
    RESULT_DIR / "mobilenetv2_combined_confusion_matrix.csv"
)

print("\nCombined training completed!")

print("Model:", BEST_MODEL_PATH)
print("Results:", RESULT_DIR)

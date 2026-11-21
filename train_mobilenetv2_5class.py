
from pathlib import Path
import random
import copy

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(r"D:\VISIONX")

DATA_DIR = ROOT / "dataset_split"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

CLASSES = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise"
]

NUM_CLASSES = 5

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 8

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

NUM_WORKERS = 0
SEED = 42

# New checkpoint: does not overwrite previous models
OUTPUT_CHECKPOINT = (
    MODEL_DIR / "mobilenetv2_5class_scratch_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

def seed_everything():

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. CLAHE PREPROCESSING
# =========================================================

class ApplyCLAHE:

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def __call__(self, image):

        image = np.asarray(image.convert("RGB"))

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        enhanced = self.clahe.apply(gray)

        rgb = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2RGB
        )

        return Image.fromarray(rgb)


# =========================================================
# 4. RANDOM OCCLUSION AUGMENTATION
# =========================================================

class RandomOcclusion:

    def __init__(self, probability=0.35):
        self.probability = probability

    def __call__(self, tensor):

        if random.random() < self.probability:

            _, height, width = tensor.shape

            box_h = random.randint(
                8, max(9, height // 4)
            )

            box_w = random.randint(
                8, max(9, width // 4)
            )

            y = random.randint(
                0, height - box_h
            )

            x = random.randint(
                0, width - box_w
            )

            tensor[
                :, y:y + box_h, x:x + box_w
            ] = 0

        return tensor


# =========================================================
# 5. DATA TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    ApplyCLAHE(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    RandomOcclusion(
        probability=0.35
    ),

    transforms.Normalize(
        MEAN,
        STD
    )
])


eval_transform = transforms.Compose([

    ApplyCLAHE(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )
])


# =========================================================
# 6. CBAM CHANNEL ATTENTION
# =========================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels=1280, reduction=16):

        super().__init__()

        hidden = max(
            channels // reduction,
            1
        )

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

        avg = self.mlp(
            self.avg_pool(x)
        )

        maximum = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg + maximum
        )

        return x * attention


# =========================================================
# 7. CBAM SPATIAL ATTENTION
# =========================================================

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

        avg = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        maximum, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg, maximum],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# =========================================================
# 8. COMPLETE CBAM
# =========================================================

class CBAM(nn.Module):

    def __init__(self, channels=1280):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=16
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# =========================================================
# 9. MOBILENETV2 + CBAM MODEL
# =========================================================

class MobileNetV2CBAM(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        # IMPORTANT:
        # weights=None means training from scratch.
        # No ImageNet pretrained weights are loaded.

        backbone = models.mobilenet_v2(
            weights=None
        )

        self.features = backbone.features

        self.cbam = CBAM(
            channels=1280
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Dropout(p=0.3),

            nn.Linear(
                1280,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x


# =========================================================
# 10. DATASET VALIDATION
# =========================================================

def load_datasets():

    expected = sorted(CLASSES)

    for split in ["train", "val", "test"]:

        path = DATA_DIR / split

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset folder missing: {path}"
            )

        actual = sorted([
            p.name.lower()
            for p in path.iterdir()
            if p.is_dir()
        ])

        if actual != expected:

            raise ValueError(
                f"\nIncorrect classes in {split}.\n"
                f"Expected: {expected}\n"
                f"Found: {actual}"
            )

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

    if (
        train_dataset.class_to_idx
        != val_dataset.class_to_idx
        or train_dataset.class_to_idx
        != test_dataset.class_to_idx
    ):
        raise ValueError(
            "Class index mapping differs between splits."
        )

    return train_dataset, val_dataset, test_dataset


# =========================================================
# 11. TRAINING AND VALIDATION FUNCTIONS
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    use_amp = DEVICE.type == "cuda"

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


def validate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            average="macro",

            zero_division=0
        )
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# =========================================================
# 12. FINAL TEST EVALUATION
# =========================================================

def evaluate_test(model, loader, class_names):

    model.eval()

    y_true = []
    y_pred = []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            y_true,

            y_pred,

            average="macro",

            zero_division=0
        )
    )

    report = classification_report(

        y_true,

        y_pred,

        labels=list(range(len(class_names))),

        target_names=class_names,

        zero_division=0
    )

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=list(range(len(class_names)))
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        report,
        matrix
    )


# =========================================================
# 13. MAIN
# =========================================================

def main():

    seed_everything()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n======================================")
    print("VISIONX - FIVE CLASS TRAINING")
    print("MobileNetV2 + CBAM")
    print("Training from scratch")
    print("======================================")

    print("\nDevice:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nLoading five-class dataset...")

    train_dataset, val_dataset, test_dataset = (
        load_datasets()
    )

    class_names = train_dataset.classes

    print("\nClasses:", class_names)

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    # Class distribution

    targets = np.array(
        train_dataset.targets
    )

    counts = np.bincount(
        targets,
        minlength=NUM_CLASSES
    )

    print("\nTraining class distribution:")

    for name, count in zip(
        class_names,
        counts
    ):

        print(f"{name}: {count}")

    # Balanced sampling

    class_weights = 1.0 / np.maximum(
        counts,
        1
    )

    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(

        weights=torch.DoubleTensor(
            sample_weights
        ),

        num_samples=len(train_dataset),

        replacement=True
    )

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=sampler,

        num_workers=NUM_WORKERS,

        pin_memory=pin_memory
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=pin_memory
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=pin_memory
    )

    # -----------------------------------------------------
    # CREATE NEW MODEL FROM SCRATCH
    # -----------------------------------------------------

    print("\nCreating new MobileNetV2 + CBAM model...")

    model = MobileNetV2CBAM(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    print("Model initialized with random weights.")
    print("No ImageNet pretrained weights loaded.")

    # Loss

    criterion = nn.CrossEntropyLoss()

    # Optimizer

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    best_val_f1 = -1.0

    best_epoch = 0

    wait = 0

    best_state = None

    # =====================================================
    # 14. TRAINING LOOP
    # =====================================================

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler
        )

        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1
        ) = validate_model(

            model,

            val_loader,

            criterion
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n--------------------------------------")

        print(
            f"Epoch [{epoch}/{EPOCHS}]"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc * 100:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_acc * 100:.2f}%"
        )

        print(
            f"Validation Precision: {val_precision * 100:.2f}%"
        )

        print(
            f"Validation Recall: {val_recall * 100:.2f}%"
        )

        print(
            f"Validation Macro F1: {val_f1 * 100:.2f}%"
        )

        # Save best model based on validation macro F1

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            wait = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save({

                "model_state_dict": best_state,

                "class_names": class_names,

                "class_to_idx": train_dataset.class_to_idx,

                "num_classes": NUM_CLASSES,

                "image_size": IMAGE_SIZE,

                "imagenet_mean": MEAN,

                "imagenet_std": STD,

                "best_epoch": best_epoch,

                "best_val_macro_f1": best_val_f1,

                "training_from_scratch": True

            }, OUTPUT_CHECKPOINT)

            print("\nBest model saved!")

        else:

            wait += 1

            print(
                f"No improvement: {wait}/{PATIENCE}"
            )

            if wait >= PATIENCE:

                print("\nEarly stopping triggered.")

                break

    # =====================================================
    # 15. LOAD BEST MODEL
    # =====================================================

    print("\nLoading best five-class model...")

    checkpoint = torch.load(

        OUTPUT_CHECKPOINT,

        map_location=DEVICE,

        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # =====================================================
    # 16. TEST EVALUATION
    # =====================================================

    print("\nEvaluating on test dataset...")

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        report,
        matrix
    ) = evaluate_test(

        model,

        test_loader,

        class_names
    )

    print("\n======================================")
    print("VISIONX FIVE-CLASS TEST RESULTS")
    print("======================================")

    print("Best epoch:", best_epoch)

    print(
        f"Test Accuracy: {test_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: {test_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall: {test_recall * 100:.2f}%"
    )

    print(
        f"Macro F1-Score: {test_f1 * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(report)

    print("\nConfusion Matrix:")

    print(matrix)

    # =====================================================
    # 17. SAVE RESULTS
    # =====================================================

    report_path = (
        RESULT_DIR /
        "visionx_5class_classification_report.txt"
    )

    matrix_path = (
        RESULT_DIR /
        "visionx_5class_confusion_matrix.csv"
    )

    metrics_path = (
        RESULT_DIR /
        "visionx_5class_metrics.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    np.savetxt(

        matrix_path,

        matrix,

        delimiter=",",

        fmt="%d"
    )

    metrics_path.write_text(

        f"Best Epoch: {best_epoch}\n"
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
        f"Macro Precision: {test_precision * 100:.2f}%\n"
        f"Macro Recall: {test_recall * 100:.2f}%\n"
        f"Macro F1: {test_f1 * 100:.2f}%\n",

        encoding="utf-8"
    )

    print("\n======================================")
    print("TRAINING AND EVALUATION COMPLETED")
    print("======================================")

    print("\nSaved checkpoint:")
    print(OUTPUT_CHECKPOINT)

    print("\nSaved classification report:")
    print(report_path)

    print("\nSaved confusion matrix:")
    print(matrix_path)

    print("\nSaved metrics:")
    print(metrics_path)


# =========================================================
# 18. START PROGRAM
# =========================================================

if __name__ == "__main__":

    main()

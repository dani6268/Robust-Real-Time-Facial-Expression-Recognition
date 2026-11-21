
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights


# ================= CONFIGURATION =================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset_aligned"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_RATIO = 0.10
SEED = 42
NUM_WORKERS = 0

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================= CLAHE =================

class ApplyCLAHE:
    def __call__(self, image):
        gray = np.asarray(image.convert("L"))

        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_GRID,
        )

        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)


# ================= OCCLUSION AUGMENTATION =================

class RandomOcclusion:
    def __init__(self, probability=0.35):
        self.probability = probability

    def __call__(self, tensor):
        if random.random() >= self.probability:
            return tensor

        _, height, width = tensor.shape

        occ_width = max(
            1, int(width * random.uniform(0.08, 0.20))
        )
        occ_height = max(
            1, int(height * random.uniform(0.08, 0.20))
        )

        left = random.randint(0, width - occ_width)
        top = random.randint(0, height - occ_height)

        tensor = tensor.clone()
        tensor[
            :, top:top + occ_height, left:left + occ_width
        ] = 0.0

        return tensor


# ================= CBAM =================

class CBAM(nn.Module):
    def __init__(self, channels=1280, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.spatial = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)

        channel_attention = torch.sigmoid(
            self.mlp(avg) + self.mlp(maximum)
        )
        x = x * channel_attention

        avg = x.mean(dim=1, keepdim=True)
        maximum = x.max(dim=1, keepdim=True)[0]

        spatial_input = torch.cat([avg, maximum], dim=1)
        spatial_attention = torch.sigmoid(
            self.spatial(spatial_input)
        )

        return x * spatial_attention


# ================= MOBILENETV2 =================

class FERMobileNetV2(nn.Module):
    def __init__(self, num_classes, use_cbam=False):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )

        self.features = backbone.features

        if use_cbam:
            self.attention = CBAM(1280)
        else:
            self.attention = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(
        description="VISIONX MobileNetV2 controlled experiments"
    )

    parser.add_argument("--name", default="experiment")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--cbam", action="store_true")
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")

    args = parser.parse_args()

    # Keep output filenames safe.
    exp_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in args.name
    )

    if not exp_name:
        raise ValueError("Please provide a valid experiment name.")

    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folders not found:\n"
            f"{TRAIN_DIR}\n{TEST_DIR}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("VISIONX CONTROLLED EXPERIMENT")
    print("=" * 55)
    print("Experiment:", exp_name)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(
        f"CLAHE={args.clahe}, CBAM={args.cbam}, "
        f"Occlusion={args.occlusion}"
    )
    print(
        f"Class weighted={args.class_weighted}, "
        f"Weighted sampler={args.weighted_sampler}"
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Training transforms
    train_ops = []

    if args.clahe:
        train_ops.append(ApplyCLAHE())

    train_ops.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
    ])

    if args.occlusion:
        train_ops.append(RandomOcclusion(probability=0.35))

    train_ops.append(transforms.Normalize(mean, std))

    # Evaluation transforms: no random augmentation
    eval_ops = []

    if args.clahe:
        eval_ops.append(ApplyCLAHE())

    eval_ops.extend([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base = datasets.ImageFolder(
        TRAIN_DIR,
        transform=transforms.Compose(train_ops),
    )

    eval_base = datasets.ImageFolder(
        TRAIN_DIR,
        transform=transforms.Compose(eval_ops),
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transforms.Compose(eval_ops),
    )

    if len(train_base) == 0 or len(test_dataset) == 0:
        raise ValueError("Training or test dataset is empty.")

    if train_base.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Training and testing classes do not match.\n"
            f"Train: {train_base.class_to_idx}\n"
            f"Test: {test_dataset.class_to_idx}"
        )

    classes = train_base.classes
    targets = np.asarray(train_base.targets)
    indices = np.arange(len(targets))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=targets,
    )

    train_dataset = Subset(
        train_base, train_indices.tolist()
    )
    val_dataset = Subset(
        eval_base, val_indices.tolist()
    )

    train_labels = targets[train_indices]
    pin_memory = device.type == "cuda"

    # Optional weighted sampler
    sampler = None
    shuffle = True

    if args.weighted_sampler:
        counts = np.bincount(
            train_labels, minlength=len(classes)
        )

        class_sample_weights = 1.0 / np.maximum(counts, 1)
        sample_weights = class_sample_weights[train_labels]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weights, dtype=torch.double
            ),
            num_samples=len(sample_weights),
            replacement=True,
        )

        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print("Classes:", classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    # Model
    model = FERMobileNetV2(
        num_classes=len(classes),
        use_cbam=args.cbam,
    ).to(device)

    # Optional class-weighted loss
    if args.class_weighted:
        counts = np.bincount(
            train_labels, minlength=len(classes)
        ).astype(np.float64)

        weights = len(train_labels) / (
            len(classes) * np.maximum(counts, 1)
        )

        weights = torch.tensor(
            weights, dtype=torch.float32, device=device
        )

        criterion = nn.CrossEntropyLoss(weight=weights)

        print(
            "Class weights:",
            {
                name: round(float(weight), 3)
                for name, weight in zip(
                    classes, weights.cpu()
                )
            },
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_accuracy = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    waiting = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    checkpoint_path = (
        MODEL_DIR / f"mobilenetv2_{exp_name}_best.pth"
    )

    # ================= TRAINING LOOP =================

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            loss_sum += loss.item() * batch_count
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            count += batch_count

        train_loss = loss_sum / count
        train_accuracy = correct / count

        # Validation
        model.eval()
        loss_sum = 0.0
        correct = 0
        count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_count = labels.size(0)
                loss_sum += loss.item() * batch_count
                correct += (
                    outputs.argmax(1) == labels
                ).sum().item()
                count += batch_count

        val_loss = loss_sum / count
        val_accuracy = correct / count

        scheduler.step(val_accuracy)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2%} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.2%} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0

            torch.save(
                {
                    "model_name": f"MobileNetV2-{exp_name}",
                    "model_state_dict": best_state,
                    "class_names": classes,
                    "class_to_idx": train_base.class_to_idx,
                    "image_size": IMAGE_SIZE,
                    "imagenet_mean": mean,
                    "imagenet_std": std,
                    "use_clahe": args.clahe,
                    "use_cbam": args.cbam,
                    "use_occlusion": args.occlusion,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_accuracy,
                },
                checkpoint_path,
            )

            print("  Saved best checkpoint.")
        else:
            waiting += 1

        if waiting >= PATIENCE:
            print("Early stopping: validation accuracy did not improve.")
            break

    # ================= TEST EVALUATION =================

    model.load_state_dict(best_state)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    test_accuracy = accuracy_score(y_true, y_pred)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    print("\n========== FINAL RESULTS ==========")
    print("Experiment:", exp_name)
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Macro precision: {precision:.2%}")
    print(f"Macro recall: {recall:.2%}")
    print(f"Macro F1-score: {macro_f1:.2%}")
    print("\nClassification report:\n")
    print(report)

    # Save results separately for each experiment
    metrics_path = (
        RESULTS_DIR / f"mobilenetv2_{exp_name}_metrics.json"
    )
    report_path = (
        RESULTS_DIR / f"mobilenetv2_{exp_name}_report.txt"
    )
    matrix_path = (
        RESULTS_DIR / f"mobilenetv2_{exp_name}_confusion_matrix.csv"
    )
    plot_path = (
        RESULTS_DIR / f"mobilenetv2_{exp_name}_curves.png"
    )

    metrics = {
        "experiment": exp_name,
        "clahe": args.clahe,
        "cbam": args.cbam,
        "occlusion": args.occlusion,
        "class_weighted": args.class_weighted,
        "weighted_sampler": args.weighted_sampler,
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(best_accuracy),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "class_names": classes,
        "history": history,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report_path.write_text(report, encoding="utf-8")

    with matrix_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + classes)

        for name, row in zip(classes, matrix):
            writer.writerow([name] + row.tolist())

    # Save training curves
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        epochs, history["train_accuracy"], label="Train"
    )
    axes[0].plot(
        epochs, history["val_accuracy"], label="Validation"
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs, history["train_loss"], label="Train"
    )
    axes[1].plot(
        epochs, history["val_loss"], label="Validation"
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print("\nSaved files:")
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)
    print("Report:", report_path)
    print("Confusion matrix:", matrix_path)
    print("Training curves:", plot_path)


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(r"D:\VISIONX")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Previously measured test accuracy values
models = [
    # MobileNet-V1 experiments
    ("Baseline MobileNet-V1", 57.73),
    ("MobileNet-V1 + CLAHE", 58.29),
    ("MobileNet-V1 + Face Alignment", 57.34),
    ("MobileNet-V1 + CBAM", 62.11),
    ("Combined MobileNet-V1", 63.04),

    # MobileNet-V2 experiments
    ("Baseline MobileNet-V2", 67.50),
    (
        "MobileNet-V2 + CLAHE + Alignment + Occlusion + CBAM",
        65.23,
    ),
]

# Print comparison table
print("\n========== VISIONX MODEL COMPARISON ==========\n")
print(f"{'Model':<55} {'Test Accuracy':>15}")
print("-" * 72)

for name, accuracy in models:
    print(f"{name:<55} {accuracy:>14.2f}%")

best_model = max(models, key=lambda item: item[1])

print("-" * 72)
print(f"Best model: {best_model[0]}")
print(f"Test accuracy: {best_model[1]:.2f}%")

# Save comparison to CSV
csv_path = RESULTS_DIR / "all_model_accuracies.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Test Accuracy (%)"])
    writer.writerows(models)

print(f"\nCSV saved to: {csv_path}")

# Create bar chart
plt.figure(figsize=(15, 8))

names = [name for name, _ in models]
accuracies = [accuracy for _, accuracy in models]

bars = plt.bar(names, accuracies)

plt.title("VISIONX - Model Test Accuracy Comparison")
plt.ylabel("Test Accuracy (%)")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

chart_path = RESULTS_DIR / "all_model_accuracies.png"
plt.savefig(chart_path, dpi=200, bbox_inches="tight")

print(f"Chart saved to: {chart_path}")

plt.show()
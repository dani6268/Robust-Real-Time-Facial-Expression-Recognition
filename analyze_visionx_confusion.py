
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    ROOT / "results" /
    "mobilenetv2_combined_confusion_matrix.csv"
)

PLOT_PATH = (
    ROOT / "results" /
    "mobilenetv2_combined_confusion_heatmap.png"
)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Confusion matrix not found: {CSV_PATH}"
        )

    with CSV_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.reader(file))

    classes = rows[0][1:]
    matrix = np.array(
        [[int(value) for value in row[1:]] for row in rows[1:]],
        dtype=int,
    )

    if matrix.shape != (len(classes), len(classes)):
        raise ValueError("Invalid confusion matrix format.")

    print("\nCONFUSION MATRIX")
    print("Rows = actual expression")
    print("Columns = predicted expression")
    print("Classes:", classes)
    print(matrix)

    errors = []

    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            if i != j and matrix[i, j] > 0:
                errors.append(
                    (matrix[i, j], actual, predicted)
                )

    errors.sort(reverse=True)

    print("\nMOST COMMON CONFUSIONS")

    for count, actual, predicted in errors[:15]:
        print(
            f"Actual {actual:10s} predicted as "
            f"{predicted:10s}: {count}"
        )

    print("\nPER-CLASS RECALL")

    for i, name in enumerate(classes):
        total = matrix[i].sum()
        recall = matrix[i, i] / total if total else 0

        print(
            f"{name:10s}: {recall:.2%} "
            f"({matrix[i, i]}/{total})"
        )

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest")

    ax.set_title("VISIONX Confusion Matrix")
    ax.set_xlabel("Predicted expression")
    ax.set_ylabel("Actual expression")
    ax.set_xticks(
        range(len(classes)), classes,
        rotation=45, ha="right"
    )
    ax.set_yticks(range(len(classes)), classes)

    fig.colorbar(image, ax=ax, label="Number of images")

    threshold = matrix.max() / 2

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print(f"\nHeatmap saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()

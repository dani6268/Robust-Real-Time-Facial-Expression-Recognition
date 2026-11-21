
import random
import shutil
from pathlib import Path

SOURCE = Path(r"D:\VISIONX\dataset_combined")
DEST = Path(r"D:\VISIONX\dataset_split")

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard class names and accepted folder names
CLASS_ALIASES = {
    "angry": ["angry"],
    "disgust": ["disgust"],
    "fear": ["fear"],
    "happy": ["happy"],
    "neutral": ["neutral"],
    "sad": ["sad"],
    "surprise": ["surprise", "suprise"],
}


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source folder not found: {SOURCE}")

    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Rename it or choose a new "
            "output path after checking its contents."
        )

    # Find class folders without depending on capitalization
    existing = {
        folder.name.lower(): folder
        for folder in SOURCE.iterdir()
        if folder.is_dir()
    }

    random_generator = random.Random(SEED)
    counts = {}

    # Validate all classes before copying anything
    class_images = {}

    for class_name, aliases in CLASS_ALIASES.items():
        folder = next(
            (existing[a] for a in aliases if a in existing),
            None,
        )

        if folder is None:
            raise FileNotFoundError(
                f"Missing folder for class '{class_name}'. "
                f"Expected one of: {aliases}"
            )

        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in {folder}: {len(images)}"
            )

        class_images[class_name] = images

    # Create a reproducible split separately for each class
    for class_name, images in class_images.items():
        random_generator.shuffle(images)

        total = len(images)
        train_end = int(total * 0.80)
        val_end = int(total * 0.90)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            target_dir = DEST / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for index, source_file in enumerate(split_images):
                # Unique filenames avoid collisions
                filename = (
                    f"{class_name}_{index:07d}{source_file.suffix.lower()}"
                )
                shutil.copy2(source_file, target_dir / filename)

            counts[(split_name, class_name)] = len(split_images)

    print("\nDataset split completed!")
    print(f"Output folder: {DEST}")

    for split_name in ["train", "val", "test"]:
        print(f"\n{split_name.upper()}")
        total = 0

        for class_name in CLASS_ALIASES:
            count = counts[(split_name, class_name)]
            print(f"  {class_name:<10} {count:>7}")
            total += count

        print(f"  {'TOTAL':<10} {total:>7}")

    print("\nOriginal dataset has not been moved or deleted.")


if __name__ == "__main__":
    main()


from pathlib import Path
import csv
import cv2
import mediapipe as mp
import numpy as np

# -------------------- CONFIGURATION --------------------

PROJECT = Path(r"D:\VISIONX")
SOURCE_DIR = PROJECT / "dataset"
OUTPUT_DIR = PROJECT / "dataset_aligned"
MODEL_PATH = PROJECT / "models" / "face_landmarker.task"

IMAGE_SIZE = 48

# First run: process only 20 images per class.
# After checking the results, change this to None for the full dataset.
MAX_IMAGES_PER_CLASS = None

# Resize images before landmark detection.
DETECTION_SIZE = 192

# -------------------------------------------------------

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Face landmark model not found: {MODEL_PATH}")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
)

# Eye-region landmark indices used by the Face Mesh topology.
LEFT_EYE_IDS = [33, 133, 159, 145]
RIGHT_EYE_IDS = [362, 263, 386, 374]


def align_face(gray, landmarker):
    """
    Returns:
        aligned image, success flag, rotation angle

    If landmarks cannot be detected or look unreliable,
    return the original image resized to the target dimensions.
    """
    original = cv2.resize(
        gray, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    enlarged = cv2.resize(
        gray, (DETECTION_SIZE, DETECTION_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    rgb = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return original, False, 0.0

    landmarks = result.face_landmarks[0]

    def eye_center(indices):
        points = np.array(
            [
                [
                    landmarks[i].x * DETECTION_SIZE,
                    landmarks[i].y * DETECTION_SIZE,
                ]
                for i in indices
            ],
            dtype=np.float32,
        )
        return points.mean(axis=0)

    eye_a = eye_center(LEFT_EYE_IDS)
    eye_b = eye_center(RIGHT_EYE_IDS)

    # Ensure the first point is on the left in image coordinates.
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a

    dx = float(eye_b[0] - eye_a[0])
    dy = float(eye_b[1] - eye_a[1])

    # Reject implausible landmark results.
    eye_distance = float(np.hypot(dx, dy))

    if dx <= 0 or eye_distance < 5:
        return original, False, 0.0

    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Avoid large rotations caused by incorrect landmarks.
    if abs(angle) > 35:
        return original, False, angle

    # Convert eye coordinates back to the original image scale.
    scale = gray.shape[1] / DETECTION_SIZE
    eye_a_original = eye_a * scale
    eye_b_original = eye_b * scale
    center = (eye_a_original + eye_b_original) / 2

    # Rotate to make the line joining the eyes horizontal.
    matrix = cv2.getRotationMatrix2D(
        tuple(center.tolist()), angle, 1.0
    )

    rotated = cv2.warpAffine(
        gray,
        matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    aligned = cv2.resize(
        rotated, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return aligned, True, angle


def main():
    stats = []
    total = 0
    aligned_count = 0
    fallback_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        for split in ["train", "test"]:
            split_dir = SOURCE_DIR / split

            if not split_dir.exists():
                print(f"Skipping missing folder: {split_dir}")
                continue

            class_dirs = sorted(
                p for p in split_dir.iterdir() if p.is_dir()
            )

            for class_dir in class_dirs:
                images = sorted(
                    p for p in class_dir.rglob("*")
                    if p.suffix.lower() in {
                        ".jpg", ".jpeg", ".png", ".bmp"
                    }
                )

                if MAX_IMAGES_PER_CLASS is not None:
                    images = images[:MAX_IMAGES_PER_CLASS]

                output_class = (
                    OUTPUT_DIR / split / class_dir.name
                )
                output_class.mkdir(parents=True, exist_ok=True)

                print(
                    f"\nProcessing {split}/{class_dir.name}: "
                    f"{len(images)} images"
                )

                for index, image_path in enumerate(images, start=1):
                    gray = cv2.imread(
                        str(image_path), cv2.IMREAD_GRAYSCALE
                    )

                    if gray is None:
                        print(f"Could not read: {image_path}")
                        stats.append([
                            split, class_dir.name, image_path.name,
                            "read_error", ""
                        ])
                        continue

                    aligned, success, angle = align_face(
                        gray, landmarker
                    )

                    # Preserve the class folder and image filename.
                    output_path = output_class / (
                        image_path.stem + ".jpg"
                    )

                    saved = cv2.imwrite(
                        str(output_path), aligned,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

                    if not saved:
                        raise RuntimeError(
                            f"Could not save: {output_path}"
                        )

                    total += 1

                    if success:
                        aligned_count += 1
                        status = "aligned"
                    else:
                        fallback_count += 1
                        status = "original_fallback"

                    stats.append([
                        split, class_dir.name, image_path.name,
                        status, round(angle, 2)
                    ])

                    if index % 10 == 0 or index == len(images):
                        print(f"  {index}/{len(images)}")

    # Save processing statistics.
    report_path = OUTPUT_DIR / "alignment_report.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with report_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "split", "class", "filename", "status", "angle_degrees"
        ])
        writer.writerows(stats)

    print("\n========== ALIGNMENT SUMMARY ==========")
    print(f"Images saved:       {total}")
    print(f"Successfully aligned: {aligned_count}")
    print(f"Fallback images:     {fallback_count}")

    if total:
        print(
            f"Alignment success rate: "
            f"{aligned_count / total * 100:.2f}%"
        )

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print("\nInspect the output before processing the full dataset.")


if __name__ == "__main__":
    main()

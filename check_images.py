from pathlib import Path
from PIL import Image

dataset = Path(r"D:\VISIONX\dataset\train")
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

checked = 0

for path in dataset.rglob("*"):
    if path.suffix.lower() not in extensions:
        continue

    try:
        with Image.open(path) as image:
            print("Sample:", path.name)
            print("Image size:", image.size)
            print("Image mode:", image.mode)
            checked += 1
            if checked == 5:
                break
    except Exception as error:
        print("Could not read:", path, error)

if checked == 0:
    print("No supported image files found.")
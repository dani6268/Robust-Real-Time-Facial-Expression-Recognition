
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from collections import deque


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(r"D:\VISIONX")

CHECKPOINT = (
    ROOT / "models" /
    "mobilenetv2_5class_combined_best.pth"
)

CASCADE_DIR = ROOT / "models" / "haarcascades"

FACE_XML = CASCADE_DIR / "haarcascade_frontalface_default.xml"
EYE_XML = CASCADE_DIR / "haarcascade_eye.xml"

CLASSES = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise"
]

IMAGE_SIZE = 224
CAMERA_INDEX = 0

SMOOTHING_FRAMES = 10
MIN_CONFIDENCE = 45.0

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. LOAD HAAR CASCADES
# ==========================================

face_cascade = cv2.CascadeClassifier(str(FACE_XML))
eye_cascade = cv2.CascadeClassifier(str(EYE_XML))

if face_cascade.empty():
    raise RuntimeError(f"Face XML not loaded: {FACE_XML}")

if eye_cascade.empty():
    raise RuntimeError(f"Eye XML not loaded: {EYE_XML}")

print("Haar Cascade files loaded successfully!")


# ==========================================
# 3. FACIAL ALIGNMENT
# Same preprocessing as training
# ==========================================

def align_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image

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

    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# ==========================================
# 4. CLAHE
# ==========================================

def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    result = cv2.merge([
        enhanced_l,
        a,
        b
    ])

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB
    )


# ==========================================
# 5. CBAM CHANNEL ATTENTION
# Must match training code
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

        attention = self.sigmoid(
            avg_out + max_out
        )

        return x * attention


# ==========================================
# 6. CBAM SPATIAL ATTENTION
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

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ==========================================
# 7. COMPLETE CBAM
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
# 8. MOBILENETV2 + CBAM
# Same architecture as training
# ==========================================

class MobileNetV2CBAM(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.backbone = models.mobilenet_v2(
            weights=None
        )

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
# 9. LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint["class_names"] != CLASSES:
        raise ValueError(
            f"Checkpoint classes: {checkpoint['class_names']}\n"
            f"Expected classes: {CLASSES}"
        )

    model = MobileNetV2CBAM(
        num_classes=len(CLASSES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("5-class MobileNetV2 + CBAM loaded!")
    print("Checkpoint:", CHECKPOINT)
    print("Best epoch:", checkpoint["best_epoch"])
    print("Device:", DEVICE)

    return model


# ==========================================
# 10. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


def preprocess_face(face_bgr):

    # BGR to RGB
    image = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Alignment
    image = align_face(image)

    # CLAHE
    image = apply_clahe(image)

    image = Image.fromarray(image)

    tensor = transform(image).unsqueeze(0)

    return tensor.to(DEVICE)


# ==========================================
# 11. MAIN WEBCAM
# ==========================================

def main():

    model = load_model()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam!")

    probability_history = deque(
        maxlen=SMOOTHING_FRAMES
    )

    print("\nVISIONX - 5 Expression Recognition")
    print("Expressions:", CLASSES)
    print("Press Q to quit.")
    print("Press R to reset smoothing.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) > 0:

                x, y, w, h = max(
                    faces,
                    key=lambda box: box[2] * box[3]
                )

                pad_x = int(w * 0.12)
                pad_y = int(h * 0.12)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)

                x2 = min(
                    frame.shape[1],
                    x + w + pad_x
                )

                y2 = min(
                    frame.shape[0],
                    y + h + pad_y
                )

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:

                    input_tensor = preprocess_face(face_crop)

                    with torch.inference_mode():

                        outputs = model(input_tensor)

                        probabilities = torch.softmax(
                            outputs,
                            dim=1
                        )[0].cpu().numpy()

                    probability_history.append(probabilities)

                    average_probs = np.mean(
                        probability_history,
                        axis=0
                    )

                    predicted_index = int(
                        np.argmax(average_probs)
                    )

                    label = CLASSES[predicted_index]

                    confidence = float(
                        average_probs[predicted_index] * 100
                    )

                    if confidence < MIN_CONFIDENCE:
                        display_text = "UNCERTAIN"
                        color = (0, 200, 255)
                    else:
                        display_text = (
                            f"{label.upper()} {confidence:.1f}%"
                        )
                        color = (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        display_text,
                        (x1, max(30, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2
                    )

                    # Top two predictions
                    top_indices = np.argsort(
                        average_probs
                    )[-2:][::-1]

                    top_text = (
                        f"{CLASSES[top_indices[0]]}: "
                        f"{average_probs[top_indices[0]] * 100:.1f}%"
                        f" | {CLASSES[top_indices[1]]}: "
                        f"{average_probs[top_indices[1]] * 100:.1f}%"
                    )

                    cv2.putText(
                        frame,
                        top_text,
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1
                    )

            else:

                probability_history.clear()

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2
                )

            cv2.putText(
                frame,
                "VISIONX | 5 Expression Recognition",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(
                "VISIONX - 5 Expression Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                probability_history.clear()
                print("Smoothing reset.")

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("VISIONX webcam closed.")


if __name__ == "__main__":
    main()

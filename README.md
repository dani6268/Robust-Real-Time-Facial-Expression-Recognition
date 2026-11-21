# VISIONX – Real-Time Facial Expression Recognition

<p align="center">
  <b>A Deep Learning-Based Facial Expression Recognition System Using MobileNetV2 and CBAM</b>
</p>

VISIONX is a computer vision and deep learning project that recognizes five facial expressions in real time using a webcam. It combines MobileNetV2 with the Convolutional Block Attention Module (CBAM) to extract important facial features and classify expressions.

The project integrates image enhancement, facial alignment, occlusion augmentation, and attention mechanisms to build a facial expression recognition system.

---

## 1. Project Overview

Facial expression recognition is an important application of computer vision and artificial intelligence. It involves analyzing facial features and classifying visible facial expressions.

VISIONX uses a trained deep learning model to recognize the following five expressions:

| Expression | Description |
|---|---|
| Angry | Facial expression associated with anger |
| Fear | Facial expression associated with fear |
| Happy | Facial expression associated with happiness |
| Sad | Facial expression associated with sadness |
| Surprise | Facial expression associated with surprise |

The system captures a face through a webcam, processes the image, and predicts the most likely expression with a confidence score.

**Note:** The system classifies visible facial expressions. It does not determine a person's actual emotional or psychological state.

---

## 2. Project Objectives

The main objectives of VISIONX are:

- Develop a real-time facial expression recognition system.
- Train a MobileNetV2 model from scratch.
- Improve image quality using CLAHE.
- Align facial images using OpenCV Haar Cascades.
- Improve robustness through occlusion augmentation.
- Integrate CBAM attention into MobileNetV2.
- Compare a baseline model with an enhanced model.
- Perform real-time predictions using a webcam.

---

## 3. System Architecture

The project uses two separate experiments.

### Experiment 1: Baseline Model

The baseline model uses MobileNetV2 without additional enhancement techniques.

**Architecture:**

Input Image → Resize → Normalization → MobileNetV2 → Global Average Pooling → Classifier → Expression Prediction

### Experiment 2: Combined Model

The combined model integrates four additional techniques with MobileNetV2.

**Architecture:**

Input Image → Facial Alignment → CLAHE → Resize → Normalization → MobileNetV2 → CBAM Attention → Global Average Pooling → Classifier → Expression Prediction

During training, Random Erasing is additionally applied as an occlusion augmentation technique.

### Combined Model Components

| Component | Purpose |
|---|---|
| MobileNetV2 | Extracts facial features |
| CLAHE | Enhances image contrast |
| Facial Alignment | Corrects face rotation using eye positions |
| Random Erasing | Simulates partial occlusion during training |
| CBAM | Focuses on important channel and spatial features |
| Softmax | Converts model outputs into class probabilities |

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image processing and MobileNetV2 |
| OpenCV | Face detection, alignment, and webcam |
| NumPy | Numerical operations |
| Pandas | Training history and result storage |
| Scikit-learn | Model evaluation and performance metrics |
| CUDA | GPU-accelerated training |

---

## 5. Understanding the Main Techniques

### 5.1 MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification.

It uses inverted residual blocks and linear bottlenecks to extract meaningful image features with relatively low computational cost.

In VISIONX:

- MobileNetV2 is used as the backbone architecture.
- Pretrained weights are not used.
- The model is trained from scratch.
- The final classification layer is modified for five expression classes.

### 5.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE improves local image contrast.

It is useful when facial images have uneven lighting or low contrast.

VISIONX applies CLAHE using the LAB color space.

Configuration:

- Clip Limit: 2.0
- Tile Grid Size: 8 × 8

The L channel is enhanced while the color channels are preserved.

### 5.3 Facial Alignment

Facial alignment helps correct face rotation before classification.

VISIONX uses OpenCV Haar Cascade classifiers to detect:

- Facial regions
- Eye positions

The angle between detected eye centers is calculated, and the image is rotated to align the eyes horizontally.

If a face or two eyes cannot be detected, the original image is returned.

### 5.4 Occlusion Augmentation

Occlusion occurs when some facial features are partially hidden.

For example, facial regions may be covered by glasses, hands, hair, or other objects.

VISIONX uses PyTorch Random Erasing to simulate partial occlusion during training.

Configuration:

- Probability: 0.35
- Scale: 0.02–0.15
- Ratio: 0.3–3.3

This encourages the model to learn from multiple facial regions instead of depending on only one region.

Random Erasing is a training augmentation; it does not detect or remove real-world occlusions.

### 5.5 CBAM (Convolutional Block Attention Module)

CBAM is an attention mechanism that helps a CNN focus on important feature information.

It consists of two modules:

**Channel Attention**

- Learns which feature channels are important.
- Uses average pooling and max pooling.
- Applies a shared MLP and sigmoid activation.

**Spatial Attention**

- Learns which spatial regions are important.
- Uses channel-wise average and maximum pooling.
- Applies a 7 × 7 convolution and sigmoid activation.

In VISIONX, CBAM is applied after the MobileNetV2 feature extractor.

---

## 6. Dataset Structure

The dataset is divided into three parts:

- Training: Used to train the model.
- Validation: Used to monitor model performance and select the best checkpoint.
- Testing: Used for final evaluation.

Expected folder structure:

```text
dataset_split/
│
├── train/
│   ├── angry/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   └── surprise/
│
├── val/
│   ├── angry/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── fear/
    ├── happy/
    ├── sad/
    └── surprise/
```

Each expression folder contains its corresponding facial images.

The class order used by the model is:

```python
CLASSES = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise"
]
```

---

## 7. Model Training Configuration

The combined model uses the following training configuration:

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 + CBAM |
| Training Method | From Scratch |
| Pretrained Weights | None |
| Number of Classes | 5 |
| Input Image Size | 224 × 224 |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 8 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Loss Function | CrossEntropyLoss |
| GPU Support | CUDA |

The best model is selected using validation macro F1-score.

Early stopping prevents unnecessary training when validation performance stops improving.

---

## 8. Model Performance

VISIONX contains two experiments:

### Baseline Model

| Metric | Result |
|---|---:|
| Test Accuracy | 83.47% |
| Macro F1-Score | 82.31% |

### Combined Model

| Metric | Result |
|---|---:|
| Test Accuracy | 84.25% |
| Macro F1-Score | 83.31% |

### Performance Comparison

| Metric | Baseline | Combined |
|---|---:|---:|
| Accuracy | 83.47% | 84.25% |
| Macro F1 | 82.31% | 83.31% |

The combined model achieved an improvement of 0.78 percentage points in test accuracy and 1.00 percentage point in macro F1-score.

These results show that the combined approach performed better on the evaluated test set.

**Important:** Both experiments should use the same independent test split for a fair comparison.

---

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd VISIONX
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install torch torchvision
```

Install the remaining packages:

```bash
python -m pip install opencv-python numpy pandas pillow scikit-learn
```

### Step 4: Download Haar Cascade Files

Create this directory:

```text
models/haarcascades/
```

Download these two OpenCV XML files into it:

- haarcascade_frontalface_default.xml
- haarcascade_eye.xml

The application uses these files for face detection and eye detection.

---

## 10. Running the Training

Make sure your dataset follows the folder structure explained above.

Run the baseline experiment:

```bash
python train_mobilenetv2_baseline.py
```

Run the combined experiment:

```bash
python train_mobilenetv2_combined.py
```

The combined model trains using MobileNetV2, CLAHE, facial alignment, Random Erasing, and CBAM.

The code uses CUDA automatically when a compatible GPU is available.

Update the project root path in the Python configuration if you are running the project from a different location.

---

## 11. Real-Time Webcam Detection

After training, the best combined checkpoint is saved at:

```text
models/
└── mobilenetv2_5class_combined_best.pth
```

Run the webcam application:

```bash
python webcam_5class.py
```

The application performs these steps:

1. Opens the webcam.
2. Detects the face using OpenCV.
3. Extracts the facial region.
4. Applies facial alignment.
5. Enhances contrast using CLAHE.
6. Resizes and normalizes the image.
7. Passes the image to the trained MobileNetV2 + CBAM model.
8. Predicts one of the five expressions.
9. Displays the prediction and confidence score.

**Webcam Controls:**

| Key | Action |
|---|---|
| Q | Quit the application |
| R | Reset prediction smoothing |

The webcam interface also displays the top predictions and their confidence scores.

---

## 12. Project Folder Structure

```text
VISIONX/
│
├── dataset_split/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── haarcascades/
│   │   ├── haarcascade_frontalface_default.xml
│   │   └── haarcascade_eye.xml
│   │
│   ├── mobilenetv2_5class_baseline_best.pth
│   └── mobilenetv2_5class_combined_best.pth
│
├── results/
│   ├── mobilenetv2_baseline_history.csv
│   ├── mobilenetv2_baseline_report.txt
│   ├── mobilenetv2_baseline_confusion_matrix.csv
│   │
│   ├── mobilenetv2_combined_history.csv
│   ├── mobilenetv2_combined_report.txt
│   └── mobilenetv2_combined_confusion_matrix.csv
│
├── train_mobilenetv2_baseline.py
├── train_mobilenetv2_combined.py
├── webcam_5class.py
│
├── requirements.txt
└── README.md
```

Dataset images and trained checkpoints can be stored separately rather than committed to GitHub.

---

## 13. Results and Evaluation

The project evaluates the models using:

**Accuracy:** Measures the percentage of correctly classified test images.

**Precision:** Measures how many predictions for a particular class are correct.

**Recall:** Measures how many actual samples of a class are correctly identified.

**F1-Score:** Combines precision and recall into a single metric.

**Confusion Matrix:** Shows correct classifications and misclassifications across the five expression classes.

Macro F1-score is especially useful because it gives equal importance to each expression class.

---

## 14. Limitations

- Recognition performance can vary with lighting conditions.
- Face detection may fail for certain poses or camera angles.
- Haar Cascade alignment is approximate and may not detect eyes reliably.
- Random Erasing simulates occlusion but does not guarantee robustness to every real-world obstruction.
- Webcam predictions may differ from offline test results.
- The model recognizes five predefined classes and cannot identify expressions outside those classes.

---

## 15. Future Improvements

- Integrate advanced facial landmark detection.
- Improve recognition under challenging lighting conditions.
- Explore additional data augmentation techniques.
- Experiment with alternative CNN architectures.
- Optimize inference speed for real-time applications.
- Develop a web-based interface for expression recognition.
- Evaluate the system on additional independent datasets.

---

## 16. Learning Outcomes

Through this project, I gained practical experience in:

- Deep learning model development using PyTorch.
- CNN architecture and MobileNetV2.
- Attention mechanisms using CBAM.
- Image enhancement using CLAHE.
- Face detection and alignment using OpenCV.
- Data augmentation and class imbalance handling.
- Model evaluation using Scikit-learn.
- GPU-based model training.
- Real-time computer vision applications.

---

## 17. Author

**Developed as a Computer Science and Engineering project.**

If you find this project useful, consider giving the repository a star!

---

## License

Add a suitable open-source license to the repository before distributing the project. Also verify the dataset's license and redistribution permissions.

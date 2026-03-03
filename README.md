# Emotion Recognition using KNN

**Group 8 – CMPT Project**

## Team Members
- Sean Wotherspoon
- Alicia Lam
- Rafi Rizco
- Ringo Kojima

---

## Project Overview

We are building a facial emotion classification system that takes a facial image as input and outputs a predicted emotion label.

Final system (Milestone 2 goal):
- Capture image (mobile/web app)
- CNN model predicts emotion
- Backend API returns label + confidence
- Progressive Web App displays result

Baseline system (Milestone 1 – Progress Submission):
- Preprocess facial image
- Flatten pixel features
- Train a K-Nearest Neighbors (KNN) classifier
- Output predicted emotion + confidence score

---

## Target Emotions

For the baseline system, we classify:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## System Architecture

### Milestone 1 (Current – KNN Baseline)

Pipeline:
```
Input Image
    ↓
OpenCV Face Detection
    ↓
Crop + Resize (48x48)
    ↓
Grayscale Conversion
    ↓
Flatten to Vector
    ↓
KNN Classifier
    ↓
Emotion Prediction + Confidence
```

Train/Validation Split:
- 75% training
- 25% validation

Metrics:
- Accuracy
- Macro F1 Score

### Milestone 2 (Final System – CNN + App)

From proposal:
- Custom CNN implemented in PyTorch
- Flask or FastAPI backend
- REST endpoint /predict
- React Progressive Web App frontend
- Real-time camera emotion detection

---

## Dataset

Primary dataset for baseline:

**FER2013 (Kaggle)**
- 48 × 48 grayscale images
- 7 emotion labels (0–6)
- Well-documented and commonly used

Additional dataset (optional later):
- Facial Expression Dataset by Aaditya Singhal

---

## Repository Structure
```
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── src/
│   ├── data/
│   ├── models/
│   │   └── knn.py
│   ├── pipeline/
│   │   └── run.py
│   └── preprocess/
│       └── face_preprocess.py
├── scripts/
├── tests/
│   ├── assets/
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run pipeline (in progress)
```bash
python -m src.pipeline.run
```

---

## Testing

We use `pytest` to validate core components of the system.

### Run all tests
```bash
python -m pytest
```

### Run a specific test file
```bash
python -m pytest tests/test_preprocess.py
```
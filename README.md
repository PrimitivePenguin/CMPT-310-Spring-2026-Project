# MoodLens

CMPT 310 Group 8 project for facial emotion recognition from uploaded face images.

## Short Overview

This repository contains an end-to-end facial emotion recognition system developed for CMPT 310 in the Spring 2026 semester at Simon Fraser University. The project explores both a classical KNN baseline and a CNN-based approach to compare different modeling strategies for facial emotion recognition, then deploys the result in a web application for local use and demonstration.

It includes:

- image preprocessing and dataset preparation code
- a KNN baseline pipeline
- a PyTorch CNN training and inference path
- a Flask backend with `/health` and `/predict` endpoints
- a React + Vite frontend for uploading an image and viewing the predicted emotion

## Team Members

- Sean Wotherspoon
- Alicia Lam
- Rafi Rizco
- Ringo Kojima

## Features

- Detects and preprocesses faces using OpenCV before classification
- Supports the seven emotion labels defined in the codebase: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`
- Builds processed NumPy datasets from raw labeled image folders
- Applies optional augmentation during dataset preprocessing
- Trains and saves a CNN model with PyTorch
- Includes a KNN-based baseline workflow for comparison and manual image testing
- Exposes backend API routes for health checks and predictions
- Provides a frontend for image upload, preview, prediction, and confidence display

## Tech Stack

- Python
- PyTorch
- scikit-learn
- OpenCV
- NumPy
- Flask + Flask-CORS
- React
- Vite
- Tailwind CSS
- pytest

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── cnn_model.pt
├── src/
│   ├── app/
│   │   ├── backend/
│   │   └── frontend/
│   ├── data/
│   ├── models/
│   ├── pipeline/
│   ├── preprocess/
│   └── training/
├── tests/
├── requirements.txt
└── README.md
```

## Setup and Installation

### Python environment

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows activation command:

```bash
venv\Scripts\activate
```

### Frontend environment

From `src/app/frontend`:

```bash
npm install
```

## Quick Start

Note: A pre-trained CNN model is included, so the web app can be used immediately.

For the fastest local setup:

1. Create and activate a virtual environment from the repository root.

```bash
python -m venv venv
source venv/bin/activate
```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. Start the backend.

```bash
python -m src.app.backend.app
```

4. In a separate terminal, move to `src/app/frontend` and install frontend dependencies.

```bash
npm install
```

5. Start the frontend.

```bash
npm run dev
```

6. Open the local frontend in a browser and upload an image to request a prediction.

## How to Run

### Backend API

From the repository root:

```bash
python -m src.app.backend.app
```

This starts the Flask app on `http://127.0.0.1:5000` with:

- `GET /health`
- `POST /predict`

The `POST /predict` endpoint accepts an uploaded image and returns a JSON response with:

- `emotion`: the predicted emotion label
- `confidence`: the model confidence score
- `source`: the prediction source string

### Frontend

From `src/app/frontend`:

```bash
npm run dev
```

The Vite dev server proxies `/health` and `/predict` to `http://127.0.0.1:5000`.

### KNN baseline pipeline

From the repository root:

```bash
python -m src.pipeline.run
```

This pipeline prepares processed data if needed, loads the dataset, and opens a local file picker for manual image prediction with the KNN baseline.

### CNN training

From the repository root:

```bash
python -m src.training.train_cnn
```

The training script saves the trained model to `models/cnn_model.pt`.

## How to Reproduce Results or Demos

### Web demo

1. Install Python dependencies from the repository root.
2. Install frontend dependencies in `src/app/frontend`.
3. Start the backend with `python -m src.app.backend.app`.
4. Start the frontend with `npm run dev`.
5. Upload a face image in the frontend and submit it for prediction.

### Model and preprocessing workflow

To regenerate processed arrays or retrain locally, the code expects labeled raw images under:

- `data/raw/train/<emotion_name>/`
- `data/raw/test/<emotion_name>/`

Where `<emotion_name>` matches the labels in `src/config.py`.

Relevant commands:

```bash
python -m src.training.train_cnn
python -m src.pipeline.run
```

## Example Output

Example JSON response from `POST /predict`:

```json
{
  "emotion": "happy",
  "confidence": 0.87,
  "source": "CNN"
}
```

## Notes / Limitations

- The trained CNN model file (`models/cnn_model.pt`) is included, so predictions can be run immediately without retraining.
- Raw training and test images are not included due to size constraints.
- Reproducing preprocessing or retraining requires the expected `data/raw/train` and `data/raw/test` directory structure.
- The backend prediction route depends on the presence of the CNN model file.
- The KNN demo uses a local `tkinter` file dialog, so it is intended for an interactive desktop environment.

## Future Improvements

- Add documented dataset sourcing and dataset preparation instructions to make full reproduction easier from a clean clone.
- Document expected Python, Node.js, and npm version requirements for a more fully specified setup process.

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from src.data.processed_data import load_data, setup_files
from src.preprocess.face_preprocess import preprocess_loaded_image
from src.config import LABELS
from src.models.knn import knn_predict, accuracy, knn_predict_image

"""
Pipeline for KNN emotion recognition with data augmentation.

Data augmentation creates 8 versions of each image:
- Original, Mirrored, Rotated (±10°), Zoomed (±10%), Shifted (±5px)

Workflow:
1. setup_files() - Process raw data and save vectors
2. load_data() - Load data (with/without augmentation)
3. Train and evaluate KNN
"""

def setup_data():
    """Create processed data files if they don't exist."""
    required_files = [
        "data/processed/train_vectors.npy",
        "data/processed/train_labels.npy",
        "data/processed/test_vectors.npy",
        "data/processed/test_labels.npy"
    ]
    
    files_exist = all(os.path.exists(f) for f in required_files)
    
    if not files_exist:
        print("Creating processed data files...")
        setup_files(train_new=False)
    else:
        print("Data files already exist.")


def main():
    # Setup data
    setup_data()
    
    # Load original data (no augmentation)
    print("\n--- Loading Original Data ---")
    X_train, y_train, X_test, y_test = load_data(augment=False)
    
    # Load augmented data
    print("\n--- Loading Augmented Data ---")
    X_train_aug, y_train_aug, X_test_aug, y_test_aug = load_data(augment=True)
    
    print(f"Original training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Augmented training data: {X_train_aug.shape}, Labels: {y_train_aug.shape}")

    print(f"Original test data: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Augmented test data: {X_test_aug.shape}, Labels: {y_test_aug.shape}")

    # Run Emotion Recognition on user selected image
    user_input = ""
    while(user_input != "q"):
        print("\n--- Emotion Recognition on User-selected Image ---\n")
        # Prompt user to select image from file explorer
        # This opens a tk dialog window but I cannot seem to be able to remove that w/o it breaking T-T
        root = tk.Tk()
        image_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpeg .png .jpg")])
        root.destroy()

        # Make prediction if image is selected
        if image_path:
            print(f"Image Selected: {image_path}")
            image = cv2.imread(image_path)
            image_vector = preprocess_loaded_image(image)
            prediction = knn_predict_image(X_train, y_train, image_vector, k=3, labels=LABELS)
            print(f"Image prediction: {LABELS[prediction]}\n")
        else:
            print("Image not found or selected.")

        # Prompt user to continue or quit loop
        user_input = input("Continue or enter (q) to quit.\n")


if __name__ == "__main__":
    main()

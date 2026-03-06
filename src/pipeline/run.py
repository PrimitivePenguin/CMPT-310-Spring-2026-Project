import os
import cv2
import numpy as np
from tqdm import tqdm
from src.preprocess.face_preprocess import setup_files, load_data, process_test
from src.config import LABELS
from src.models.knn import knn_predict, accuracy

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
        setup_files(train_new=False, mirror=True, rotation=True, zoom=True, shifting=True)
    else:
        print("Data files already exist.")


def evaluate_knn(X_train, y_train, X_test, y_test, name="", k=3):
    """Train KNN and evaluate."""
    print(f"\n{name}")
    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    
    Y_pred = knn_predict(X_train, y_train, X_test, k, LABELS)
    test_acc = accuracy(y_test, Y_pred)
    
    print(f"k={k}: Accuracy = {test_acc:.4f}")
    return test_acc


def main():
    # Setup data
    setup_data()
    
    # Test on single image
    print("\n--- Testing Single Image ---")
    test_image_path = r"data\raw\test\angry\PrivateTest_88305.jpg"
    test_image = cv2.imread(test_image_path)
    if test_image is not None:
        process_test(test_image)
    
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
    
    # Evaluate
    print("\n--- Evaluation ---")
    acc_orig = evaluate_knn(X_train, y_train, X_test, y_test, 
                            name="Original Data", k=3)
    print(f"Original Accuracy: {acc_orig:.4f}")
    # Don't do this, it takes like 20 hours
    # acc_aug = evaluate_knn(X_train_aug, y_train_aug, X_test_aug, y_test_aug, 
    #                        name="Augmented Data", k=3)
    
    # # Compare
    # print(f"\nImprovement: {(acc_aug - acc_orig):.4f} ({(acc_aug / acc_orig - 1) * 100:.2f}%)")


if __name__ == "__main__":
    main()
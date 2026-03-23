import os

import numpy as np

from src.pipeline.preprocess_dataset import preprocess_dataset


PROCESSED_DIR = "data/processed"
TRAIN_VECTORS_PATH = os.path.join(PROCESSED_DIR, "train_vectors.npy")
TRAIN_LABELS_PATH = os.path.join(PROCESSED_DIR, "train_labels.npy")
TEST_VECTORS_PATH = os.path.join(PROCESSED_DIR, "test_vectors.npy")
TEST_LABELS_PATH = os.path.join(PROCESSED_DIR, "test_labels.npy")


def save_preprocessed_data(X_train, y_train, X_test, y_test):
    """
    Save preprocessed train and test arrays to disk.

    Parameters:
        X_train: Training feature vectors.
        y_train: Training labels.
        X_test: Test feature vectors.
        y_test: Test labels.
    Returns:
        None
    Raises:
        OSError: If the processed-data directory or files cannot be written.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    np.save(TRAIN_VECTORS_PATH, X_train)
    np.save(TRAIN_LABELS_PATH, y_train)
    np.save(TEST_VECTORS_PATH, X_test)
    np.save(TEST_LABELS_PATH, y_test)


def load_data(augment=False):
    """
    Load preprocessed train and test arrays from disk.

    Parameters:
        augment (bool): Whether to return the full augmented dataset.
    Returns:
        tuple: Loaded train/test vectors and labels.
    Raises:
        FileNotFoundError: If one or more processed files are missing.
    """
    X_train = np.load(TRAIN_VECTORS_PATH)
    y_train = np.load(TRAIN_LABELS_PATH)
    X_test = np.load(TEST_VECTORS_PATH)
    y_test = np.load(TEST_LABELS_PATH)

    if not augment:
        X_train = X_train[::8]
        y_train = y_train[::8]
        X_test = X_test[::8]
        y_test = y_test[::8]

    return X_train, y_train, X_test, y_test


def setup_files(train_new=False, augment_config=None):
    """
    Ensure processed data files exist and return loaded arrays.

    Parameters:
        train_new (bool): Whether to force regeneration of processed files.
        augment_config (dict | None): Augmentation settings for regeneration.
    Returns:
        tuple: Loaded train/test vectors and labels.
    Raises:
        OSError: If processed files cannot be read or written.
    """
    paths = (
        TRAIN_VECTORS_PATH,
        TRAIN_LABELS_PATH,
        TEST_VECTORS_PATH,
        TEST_LABELS_PATH,
    )
    files_exist = all(os.path.exists(path) and os.path.getsize(path) > 0 for path in paths)

    if files_exist and not train_new:
        return (
            np.load(TRAIN_VECTORS_PATH),
            np.load(TRAIN_LABELS_PATH),
            np.load(TEST_VECTORS_PATH),
            np.load(TEST_LABELS_PATH),
        )

    if augment_config is None:
        augment_config = {
            "mirror": True,
            "rotation": True,
            "zoom": True,
            "shifting": True,
        }

    X_train, y_train = preprocess_dataset("data/raw/train", augment_config=augment_config)
    X_test, y_test = preprocess_dataset("data/raw/test", augment_config=augment_config)
    save_preprocessed_data(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test

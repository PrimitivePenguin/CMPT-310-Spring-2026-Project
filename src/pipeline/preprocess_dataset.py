import os

import numpy as np

from src.config import LABELS
from src.preprocess.augment import apply_augmentation
from src.preprocess.face_preprocess import load_image, preprocess_loaded_image


def preprocess_dataset(data_dir, augment_config=None, debug=False):
    """
    Preprocess all labeled images in a dataset directory.

    Parameters:
        data_dir: Root directory containing per-label subdirectories.
        augment_config: Augmentation flags keyed by augmentation name.
        debug: Whether to print progress details.
    Returns:
        tuple[np.ndarray, np.ndarray]: Feature vectors and labels.
    Raises:
        None
    """
    if augment_config is None:
        augment_config = {
            "mirror": False,
            "rotation": False,
            "zoom": False,
            "shifting": False,
        }

    X = []
    y = []

    for emotion_label, emotion_name in enumerate(LABELS):
        emotion_dir = os.path.join(data_dir, emotion_name)

        if not os.path.exists(emotion_dir):
            if debug:
                print(f"{emotion_dir} does not exist, skipping.")
            continue

        for img_file in os.listdir(emotion_dir):
            if img_file.startswith(".") or not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                if debug:
                    print(f"Skipping non-image file: {img_file}")
                continue

            img_path = os.path.join(emotion_dir, img_file)

            try:
                img = load_image(img_path)
                augmented_imgs = apply_augmentation(img, augment_config)

                for aug_img in augmented_imgs:
                    if aug_img is None:
                        continue

                    vector = preprocess_loaded_image(aug_img)
                    X.append(vector)
                    y.append(emotion_label)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if debug:
            print(f"Finished processing {emotion_dir}, total vectors so far: {len(X)}")

    return np.array(X), np.array(y)

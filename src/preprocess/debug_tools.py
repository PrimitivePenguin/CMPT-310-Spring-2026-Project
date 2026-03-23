import os

import numpy as np

from src.preprocess.augment import mirror_image, rotate_image, shift_image, zoom_image
from src.preprocess.face_preprocess import CASCADE_PATH, preprocess_loaded_image


def debug_image_path(img_path: str) -> None:
    """
    Print basic diagnostics for an image path and cascade configuration.

    Parameters:
        img_path (str): Path to the image file being inspected.
    Returns:
        None
    Raises:
        OSError: If the file exists but cannot be opened for reading.
    """
    print(f"Cascade path: {CASCADE_PATH}")
    print(f"Image path: {img_path}")
    print(f"Exists: {os.path.exists(img_path)}")
    print(f"Is file: {os.path.isfile(img_path)}")

    if os.path.exists(img_path):
        print(f"File size: {os.path.getsize(img_path)} bytes")
        with open(img_path, "rb") as file_obj:
            header = file_obj.read(10)
        print(f"Header bytes: {header}")


def process_test(img) -> None:
    """
    Print preprocessing stats for an image and its common augmentations.

    Parameters:
        img: Input image array.
    Returns:
        None
    Raises:
        ValueError: If the input image is None.
    """
    if img is None:
        raise ValueError("Image is None.")

    augmentations = {
        "default": img,
        "mirror": mirror_image(img),
        "rotate_left": rotate_image(img, -10),
        "rotate_right": rotate_image(img, 10),
        "zoom_in": zoom_image(img, 1.1),
        "zoom_out": zoom_image(img, 0.9),
        "shift_bottom_right": shift_image(img, -5, -5),
        "shift_top_left": shift_image(img, 5, 5),
    }

    print(f"\n{'Augmentation':<20} {'Shape':<15} {'Min':<10} {'Max':<10}")
    print("-" * 55)

    for name, aug_img in augmentations.items():
        vector = preprocess_loaded_image(aug_img)
        print(
            f"{name:<20} {str(vector.shape):<15} "
            f"{np.min(vector):<10.4f} {np.max(vector):<10.4f}"
        )

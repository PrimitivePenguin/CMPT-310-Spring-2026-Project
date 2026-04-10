import os

import cv2
import numpy as np

from src.config import IMAGE_SIZE

import tempfile
import shutil

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

_tmp_dir = tempfile.mkdtemp()
_tmp_cascade = os.path.join(_tmp_dir, "haarcascade_frontalface_default.xml")
shutil.copy2(CASCADE_PATH, _tmp_cascade)

face_cascade = cv2.CascadeClassifier(_tmp_cascade)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load cascade from: {CASCADE_PATH}")


def detect_largest_face(grayscale_img: np.ndarray):
    """
    Detect the largest face in a grayscale image.

    Parameters:
        grayscale_img (np.ndarray): A grayscale image array.
    Returns:
        tuple[int, int, int, int] | None: Bounding box of the largest face, or
            None if no face is detected.
    """
    faces = face_cascade.detectMultiScale(
        grayscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None

    return tuple(max(faces, key=lambda rect: rect[2] * rect[3]))


def load_image(img_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Parameters:
        img_path (str): Path to the image file.
    Returns:
        np.ndarray: The loaded BGR image.
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If OpenCV fails to load the image.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}.")

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Could not load image from path: {img_path}.")

    return img


def preprocess_loaded_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess an already-loaded image into a normalized flattened vector.

    Parameters:
        img (np.ndarray): Input image in BGR format.
    Returns:
        np.ndarray: A 1D flattened array of shape (IMAGE_SIZE * IMAGE_SIZE,).
    Raises:
        ValueError: If the input image is None.
    """
    if img is None:
        raise ValueError("Image is None.")

    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_box_tuple = detect_largest_face(grayscale_img)

    if face_box_tuple is not None:
        x, y, w, h = face_box_tuple
        cropped_img = grayscale_img[y:y + h, x:x + w]
    else:
        cropped_img = grayscale_img

    resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))
    normalized_img = resized_img.astype(np.float32) / 255.0
    return normalized_img.flatten()


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess an image from disk into a normalized flattened vector.

    Parameters:
        img_path (str): File path of the image to preprocess.
    Returns:
        np.ndarray: A 1D flattened array of shape (IMAGE_SIZE * IMAGE_SIZE,).
    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be loaded.
    """
    img = load_image(img_path)
    return preprocess_loaded_image(img)

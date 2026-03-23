import os

import cv2
import numpy as np

from src.config import IMAGE_SIZE


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Failed to load cascade from: {CASCADE_PATH}")


def detect_largest_face(grayscale_img: np.ndarray):
    """
    Detect faces and return the bounding box of the largest face.

    Parameters:
        grayscale_img (np.ndarray): Grayscale image array.

    Returns:
        tuple[int, int, int, int] | None:
            (x, y, w, h) for the largest detected face, or None if no face found.
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
        img_path (str): Path to the image.

    Returns:
        np.ndarray: Loaded color image.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot load the image.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}.")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {img_path}.")

    return img


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load image, detect face, crop, resize, normalize, and flatten.

    Parameters:
        img_path (str): File path of the image to preprocess.

    Returns:
        np.ndarray: Flattened vector of shape (IMAGE_SIZE * IMAGE_SIZE,).
    """
    img = load_image(img_path)

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
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
    Detect faces in a grayscale image and return the bounding box of the largest face.

    This function uses OpenCV's Haar Cascade classifier to detect all faces in the
    input image and selects the face with the largest area (width × height).

    Parameters:
        grayscale_img (np.ndarray): A 2D NumPy array of shape (height, width)
                                   representing a grayscale image.

    Returns:
        tuple[int, int, int, int] | None:
            A tuple (x, y, w, h) representing the bounding box of the largest face,
            or None if no faces are detected.
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
    Load an image from disk using OpenCV.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: A color image in BGR format.

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

    This function performs:
    - grayscale conversion
    - face detection and cropping (largest face)
    - resizing to (IMAGE_SIZE × IMAGE_SIZE)
    - normalization to [0, 1]
    - flattening into a 1D vector

    This is useful for pipelines where images are already loaded (e.g., augmentation).

    Parameters:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: A 1D flattened array of shape (IMAGE_SIZE * IMAGE_SIZE,).

    Raises:
        ValueError: If the input image is None.
    """
    if img is None:
        raise ValueError("Image is None.")

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect largest face
    face_box_tuple = detect_largest_face(grayscale_img)

    if face_box_tuple is not None:
        x, y, w, h = face_box_tuple
        cropped_img = grayscale_img[y:y + h, x:x + w]
    else:
        # Fallback: use full image if no face detected
        cropped_img = grayscale_img

    # Resize to model input size
    resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize and flatten
    normalized_img = resized_img.astype(np.float32) / 255.0
    return normalized_img.flatten()


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess an image from disk into a normalized flattened vector.

    This is the primary entry point for preprocessing in the pipeline.
    It combines image loading and preprocessing into a single step.

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
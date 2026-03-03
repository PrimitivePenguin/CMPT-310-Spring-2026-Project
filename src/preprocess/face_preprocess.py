import cv2
import numpy as np
import os
from src.config import IMAGE_SIZE


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_largest_face(grayscale_img):
    """
    Detect faces and return bounding box of largest face.

    Parameters:
        grayscale_img (np.ndarray): A 2D NumPy array of shape (height, width) 
                                    containing grayscale pixel values.

    Returns:
        tuple (x, y, w, h): Bounding box coordinates if face found.
        None: If no face found.
    """
    faces = face_cascade.detectMultiScale(
        grayscale_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    if len(faces) == 0: return None

    # return largest face
    return max(faces, key=lambda rect: rect[2] * rect[3])


def preprocess_image(img_path):
    """
    Load image, detect face, crop, resize, normalize, and flatten.

    Parameters:
        img_path (str): The file path of the image to preprocess.

    Returns:
        np.ndarray: A 1D flattened array of shape (IMAGE_SIZE * IMAGE_SIZE,).
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}.")

    # 1. load image
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Could not load image.")

    # 2. convert to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 3. detect face
    face_box_tuple = detect_largest_face(grayscale_img)

    if face_box_tuple is not None:
        # use bounding box dimensions from largest face detection
        x, y, w, h = face_box_tuple
        cropped_img = grayscale_img[y:y+h, x:x+w]
    else:
        # otherwise use full image
        cropped_img = grayscale_img

    # 4. resize
    resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))

    # 5. normalize and flatten
    normalized_img = resized_img.astype(np.float32) / 255.0
    return normalized_img.flatten()
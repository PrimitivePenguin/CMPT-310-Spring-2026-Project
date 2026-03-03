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
    faces = face_cascade(
        grayscale_img,
        scaleFactor=1.1,
        minNeighbours=5,
        minSize=(30,30)
    )

    if len(faces) == 0: return None

    # return largest face
    return max(faces, key=lambda rect: rect[2] * rect[3])

def preprocess_image(image_path):
    """
    Load image, detect face, crop, resize, normalize, and flatten.

    Parameters:
        image_path (str): The file path of the image to preprocess.

    Returns:
        np.ndarray: A 1D flattened array of shape (IMAGE_SIZE * IMAGE_SIZE,).
    """
    
import cv2
import cv2.data
import numpy as np
import os
from src.config import IMAGE_SIZE
from src.config import LABELS, LABEL_TO_ID, ID_TO_LABEL

print(cv2.data.haarcascades)  # Debug: Print the path to Haar cascades
CASCADE_PATH = "venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml" # I have japanese characters in path so it causes issue so i'm using relative path instead
# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"Failed to load cascade from: {CASCADE_PATH}")


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
    # debug
    # print(f"Preprocessing image at path: {img_path}")
    # print(f"File exists: {os.path.exists(img_path)}")
    # print(f"Is file: {os.path.isfile(img_path)}")
    # if os.path.exists(img_path):
    #     print(f"File size: {os.path.getsize(img_path)} bytes")
    
    with open(img_path, 'rb') as f:
        header = f.read(10)
        # print(f"File header bytes: {header}")


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

def setup_files(reset=False):
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        "train_vectors": os.path.join(output_dir, "train_vectors.npy"),
        "train_labels": os.path.join(output_dir, "train_labels.npy"),
        "test_vectors": os.path.join(output_dir, "test_vectors.npy"),
        "test_labels": os.path.join(output_dir, "test_labels.npy")
    }
    
    if reset:
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)
    
    # Load data only if files exist and are not empty
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths.values()):
        print("Loading existing preprocessed data...")
        X_train = np.load(paths["train_vectors"])
        y_train = np.load(paths["train_labels"])
        X_test = np.load(paths["test_vectors"])
        y_test = np.load(paths["test_labels"])
    else:
        print("No valid preprocessed data found. Need to preprocess first.")
        X_train = None
        y_train = None
        X_test = None
        y_test = None
    
    return X_train, y_train, X_test, y_test

def preprocess(data_dir, debug=False):
    # Load all images from data_dir, preprocess them, and save vectors and labels to .npy files
    X_train = []
    y_train = []

    # Loop through each label directiory as emotion
    for emotion_label, emotion_name in enumerate(LABELS):
        emotion_dir = os.path.join(data_dir, emotion_name)
        if debug:
            if not os.path.exists(emotion_dir):
                print(f"{emotion_dir} does not exist, skipping.")
                continue

        # Loop through each image in emotion directory
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            try:
                vector = preprocess_image(img_path)
                X_train.append(vector)
                y_train.append(emotion_label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        if debug:
            print(f"Finished processing: {emotion_dir} total image processed: {len(X_train)}")
    print(f"Finished processing all emotions in {emotion_dir}. \n Total images processed: {len(X_train)}\n Train(shape): {len(X_train)} x {len(X_train[0])} \n labels(shape): {len(y_train)}")
    return np.array(X_train), np.array(y_train)

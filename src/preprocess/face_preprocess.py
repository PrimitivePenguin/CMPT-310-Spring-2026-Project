import cv2
import cv2.data
import numpy as np
import os
from src.config import IMAGE_SIZE
from src.config import LABELS, LABEL_TO_ID, ID_TO_LABEL
from tqdm import tqdm

print(cv2.data.haarcascades)  # Debug: Print the path to Haar cascades
CASCADE_PATH = "venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml" # I have japanese characters in path so it causes issue so i'm using relative path instead
# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"Failed to load cascade from: {CASCADE_PATH}")



""" 
Helper functions to preprocess()
 - preprocess_image(): grayscale, detect face, crop to face, resize to 48x48, normalize, and flatten an image array
 - detect_largest_face(): detects faces and returns bounding box of largest face, if no face is found return None
 Data augmentation functions (takes in color image):
 - mirror_image(): horizontally flip the image
 - rotate_image(): rotate the image by a given angle
 - zoom_image(): zoom in or out by a given factor
 - shift_image(): shift the image by given x and y pixels
"""


def preprocess_image(img):
    """Preprocess an already-loaded image array."""
    if img is None:
        raise ValueError("Image is None")
    
    # 1. convert to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. detect face
    face_box_tuple = detect_largest_face(grayscale_img)
    
    if face_box_tuple is not None:
        x, y, w, h = face_box_tuple
        cropped_img = grayscale_img[y:y+h, x:x+w]
    else:
        cropped_img = grayscale_img
    
    # 3. resize
    resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 4. normalize and flatten
    normalized_img = resized_img.astype(np.float32) / 255.0
    return normalized_img.flatten()

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
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # return largest face (faces is an array of [x, y, w, h])
    largest_face = max(faces, key=lambda face: face[2] * face[3])
    return tuple(largest_face)  # Convert to tuple (x, y, w, h)


# Data Augmentation functions

def mirror_image(img):    
    if img is None:
        return None
    return cv2.flip(img, 1)

def rotate_image(img, angle):
    if img is None:
        print("Image is None, skipping rotation.")
        return None
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h))

def zoom_image(img, zoom_factor):  # 0.9 to 1.1
    if img is None:
        return None
    
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (new_w, new_h))

    if zoom_factor > 1.0:
        # Crop center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h+h, start_w:start_w+w]
    else:
        # pad with zero
        padded = np.zeros((h, w, img.shape[2]), dtype=resized.dtype)
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        padded[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        return padded

def shift_image(img, x_shift, y_shift):
    if img is None:
        return None
    
    h, w = img.shape[:2]
    shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(img, shift_matrix, (w, h))
    return shifted


def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}.")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {img_path}.")
    return img

def apply_augmentation(img, img_path, config):
    augmented_img = [img] # default image
    if config["mirror"]:
        augmented_img.append(mirror_image(img))
    if config["rotation"]:
        augmented_img.append(rotate_image(img, angle=10)) # rotate right
        augmented_img.append(rotate_image(img, angle=-10)) # rotate left
    if config["zoom"]:
        augmented_img.append(zoom_image(img, zoom_factor=1.1)) # zoom in
        augmented_img.append(zoom_image(img, zoom_factor=0.9)) # zoom out
    if config["shifting"]:
        augmented_img.append(shift_image(img, x_shift=5, y_shift=5)) # shift top left
        augmented_img.append(shift_image(img, x_shift=-5, y_shift=-5)) # shift bottom right
    return augmented_img

# Preprocess: load images from various folders -> preprocess (and augment) -> save as .npy files vectors
def preprocess(data_dir, augment_config = None, debug=False):
    # Load all images from data_dir, preprocess them, and save vectors and labels to .npy files
    if augment_config is None:
        augment_config = {
            "mirror": False,
            "rotation": False,
            "zoom": False,
            "shifting": False
        }

    X_train = []
    y_train = []

    # Loop through each label directiory as emotion
    for emotion_label, emotion_name in enumerate(LABELS):
        emotion_dir = os.path.join(data_dir, emotion_name)

        if not os.path.exists(emotion_dir):
            if debug:
                print(f"{emotion_dir} does not exist, skipping.")
                continue
            
        # Loop through each image in emotion directory
        for img_file in os.listdir(emotion_dir):            
            if img_file.startswith('.') or not img_file.endswith(('.jpg', '.jpeg', '.png')):
                if debug:
                    print(f"Skipping non-image file: {img_file}")
                continue

            # Load image
            img_path = os.path.join(emotion_dir, img_file)
            img = load_image(img_path)  # Use the new load_image function with error handling

            if img is None:
                print(f"Warning: Failed to load image at path: {img_path}. Skipping this image.")
                continue
            
            try:
                # default image
                augmented_imgs = apply_augmentation(img, img_path, augment_config)

                for aug_img in augmented_imgs:
                    if aug_img is not None:
                        img_vector = preprocess_image(aug_img)
                        X_train.append(img_vector)
                        y_train.append(emotion_label)
                    else:
                        print(f"Augmented image is None for {img_path}, skipping this augmentation.")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        if debug:
            print(f"Finished processing: {emotion_dir} total image processed: {len(X_train)}")
    print(f"Finished processing all emotions in {emotion_dir}. \n Total images processed: {len(X_train)}\n Train(shape): {len(X_train)} x {len(X_train[0])} \n labels(shape): {len(y_train)}")
    return np.array(X_train), np.array(y_train)

# setup files

def load_data(augment = False):
    # each image has 1 original, 7 augmented (1 mirror, 2 rotation, 2 zoom, 2 shifting), total 8
    # if augment is false, only load original images (every 8th sample)
    X_train = np.load("data/processed/train_vectors.npy")
    y_train = np.load("data/processed/train_labels.npy")
    X_test = np.load("data/processed/test_vectors.npy")
    y_test = np.load("data/processed/test_labels.npy")
    
    if not augment:
        # Keep only original images (every 8th sample)
        # Pattern: [original, mirror, rotate_left, rotate_right, zoom_in, zoom_out, shift_a, shift_b, original, ...]
        X_train = X_train[::8]  # Every 8th
        y_train = y_train[::8]
        X_test = X_test[::8]
        y_test = y_test[::8]
    
    return X_train, y_train, X_test, y_test


# Setup_files() and helper functions

def save_preprocessed_data(X_train, y_train, X_test, y_test):
    """Save preprocessed arrays to disk."""
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "train_vectors.npy"), X_train)
    np.save(os.path.join(output_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(output_dir, "test_vectors.npy"), X_test)
    np.save(os.path.join(output_dir, "test_labels.npy"), y_test)
    print("Data saved successfully!")


def setup_files(train_new=False, augment_config=None):
    """Setup preprocessed data files."""
    output_dir = "data/processed"
    paths = {
        "train_vectors": os.path.join(output_dir, "train_vectors.npy"),
        "train_labels": os.path.join(output_dir, "train_labels.npy"),
        "test_vectors": os.path.join(output_dir, "test_vectors.npy"),
        "test_labels": os.path.join(output_dir, "test_labels.npy")
    }
    
    # Check if files exist and are valid
    files_exist = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths.values())
    
    if files_exist and not train_new:
        print("Loading existing preprocessed data...")
        X_train = np.load(paths["train_vectors"])
        y_train = np.load(paths["train_labels"])
        X_test = np.load(paths["test_vectors"])
        y_test = np.load(paths["test_labels"])
    else:
        print("Creating new preprocessed data...")
        if augment_config is None:
            augment_config = {"mirror": True, "rotation": True, "zoom": True, "shifting": True}
        
        train_dir = "data/raw/train"
        test_dir = "data/raw/test"
        
        X_train, y_train = preprocess(train_dir, augment_config)
        X_test, y_test = preprocess(test_dir, augment_config)
        
        save_preprocessed_data(X_train, y_train, X_test, y_test)
    
    return X_train, y_train, X_test, y_test


# test image 
def process_test(img): # Test image with all augmentations
    if img is None:
        print("Error: Image is None")
        return
    
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
        if aug_img is None:
            print(f"{name:<20} {'None':<15} {'N/A':<10} {'N/A':<10}")
            continue
        
        try:
            vector = preprocess_image(aug_img)
            print(f"{name:<20} {str(vector.shape):<15} {vector.min():<10.4f} {vector.max():<10.4f}")
        except Exception as e:
            print(f"{name:<20} {'Error':<15} {str(e)}")
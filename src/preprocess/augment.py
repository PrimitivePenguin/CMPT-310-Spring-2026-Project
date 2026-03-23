import cv2
import numpy as np


def mirror_image(img):
    """
    Create a horizontally mirrored copy of an image.

    Parameters:
        img: Input image array.
    Returns:
        np.ndarray | None: Mirrored image, or None if input is None.
    Raises:
        None
    """
    if img is None:
        return None
    return cv2.flip(img, 1)


def rotate_image(img, angle):
    """
    Rotate an image around its center.

    Parameters:
        img: Input image array.
        angle: Rotation angle in degrees.
    Returns:
        np.ndarray | None: Rotated image, or None if input is None.
    Raises:
        None
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h))


def zoom_image(img, zoom_factor):
    """
    Zoom an image in or out while preserving its original size.

    Parameters:
        img: Input image array.
        zoom_factor: Multiplicative zoom factor.
    Returns:
        np.ndarray | None: Zoomed image, or None if input is None.
    Raises:
        None
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (new_w, new_h))

    if zoom_factor > 1.0:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h + h, start_w:start_w + w]

    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if channels == 1:
        padded = np.zeros((h, w), dtype=resized.dtype)
    else:
        padded = np.zeros((h, w, channels), dtype=resized.dtype)

    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    padded[start_h:start_h + new_h, start_w:start_w + new_w] = resized
    return padded


def shift_image(img, x_shift, y_shift):
    """
    Shift an image by the requested pixel offsets.

    Parameters:
        img: Input image array.
        x_shift: Horizontal pixel offset.
        y_shift: Vertical pixel offset.
    Returns:
        np.ndarray | None: Shifted image, or None if input is None.
    Raises:
        None
    """
    if img is None:
        return None

    h, w = img.shape[:2]
    shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(img, shift_matrix, (w, h))


def apply_augmentation(img, config):
    """
    Generate configured augmented variants of an image.

    Parameters:
        img: Input image array.
        config: Augmentation flags keyed by augmentation name.
    Returns:
        list: Original image followed by enabled augmentations.
    Raises:
        None
    """
    augmented_imgs = [img]

    if config.get("mirror", False):
        augmented_imgs.append(mirror_image(img))

    if config.get("rotation", False):
        augmented_imgs.append(rotate_image(img, 10))
        augmented_imgs.append(rotate_image(img, -10))

    if config.get("zoom", False):
        augmented_imgs.append(zoom_image(img, 1.1))
        augmented_imgs.append(zoom_image(img, 0.9))

    if config.get("shifting", False):
        augmented_imgs.append(shift_image(img, 5, 5))
        augmented_imgs.append(shift_image(img, -5, -5))

    return augmented_imgs

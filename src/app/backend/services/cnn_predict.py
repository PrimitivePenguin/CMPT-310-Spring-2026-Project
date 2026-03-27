import torch
import torch.nn as nn
import numpy as np
import cv2
from src.preprocess.face_preprocess import preprocess_loaded_image
from src.config import LABELS, IMAGE_SIZE


def cnn_predict(model, image_file) -> dict:
    """
    Predict emotion and confidence of an image using a CNN model.

    Parameters:
        model: The loaded CNN model.
        image_file: The image file.
    Returns:
        dict: Prediction results. {enotion label, confidence, source}
    """
    # Handle image file (FileStorage -> np.array -> tensor)
    image_bytes = image_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    processed_image = preprocess_loaded_image(image)
    processed_image_tensor = torch.tensor(processed_image, dtype=torch.float32).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    
    # Compute CNN prediction results
    logits = model(processed_image_tensor)
    probability = nn.Softmax(dim=1)(logits)
    confidence, pred_emotion = probability.topk(1, dim=1)

    # print("--- Prediction Results ---")    
    # for i, label in enumerate(LABELS):
    #     print(f" - {label}: {probability[0][i]:.4f}")
    
    return {
        "emotion": LABELS[pred_emotion],
        "confidence": confidence.item(),
        "source": "CNN",
    }
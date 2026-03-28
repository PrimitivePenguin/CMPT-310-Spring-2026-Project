from flask import current_app as app
import os
import torch
from src.training.train_cnn import EmotionClassifierCNN
from src.config import LABELS, CNN_MODEL_PATH

def load_cnn():
    """
    Load an existing CNN model.

    Parameters:
        None
    Returns:
        EmotionClassifierCNN | None: The loaded model for evaluation or None if unable to load model.
    """
    app.logger.info("Loading CNN model...")
    model_exists = os.path.exists(CNN_MODEL_PATH)

    if not model_exists:
        app.logger.error(f"Unable to find CNN model at \"{CNN_MODEL_PATH}\"")
        return None

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)

    try:
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False))
    except:
        app.logger.error("Unable to load CNN model state")
        return None

    model.eval()

    app.logger.info("CNN model loaded")
    return model
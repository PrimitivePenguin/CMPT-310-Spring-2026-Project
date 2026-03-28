from flask import current_app as app
import os
import torch
from src.training.train_cnn import EmotionClassifierCNN
from src.config import LABELS

def load_cnn(model_file):
    """
    Load an existing CNN model.

    Parameters:
        model_path (str): Path to the trained CNN model file
    Returns:
        EmotionClassifierCNN | None: The loaded model for evaluation or None if unable to load model.
    """
    app.logger.info("Loading CNN model...")
    model_exists = os.path.exists(model_file)

    if not model_exists:
        app.logger.error(f"Unable to find CNN model at \"{model_file}\"")
        return None

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)

    try:
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=False))
    except Exception as e:
        app.logger.error("Unable to load CNN model state")
        app.logger.error(e)
        return None

    model.eval()

    app.logger.info("CNN model loaded")
    return model
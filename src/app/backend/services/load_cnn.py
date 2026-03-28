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
        EmotionClassifierCNN | None: The loaded model for evaluation or None if model is not found.
    """
    
    model_exists = os.path.exists(CNN_MODEL_PATH)

    if not model_exists:
        return None

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    return model
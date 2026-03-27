import os
import torch
from src.training.train_cnn import EmotionClassifierCNN
from src.config import LABELS

def load_cnn():
    model_path = "models/cnn_model.pt"
    
    model_exists = os.path.exists(model_path)

    if not model_exists:
        return None

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = False))
    model.eval()

    return model
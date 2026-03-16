import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.preprocess.face_preprocess import setup_files, load_data
from src.config import IMAGE_SIZE, LABELS, RAW_TRAIN_DIR, OUTPUT_FEATURE_SIZE

class EmotionClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            #images come in as 1x48x48, after conv and pooling they will be 32x24x24
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #images come in as 32x24x24, after conv and pooling they will be 64x12x12
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #images come in as 64x12x12, after conv and pooling they will be 128x6x6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, OUTPUT_FEATURE_SIZE),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(OUTPUT_FEATURE_SIZE, len(LABELS))
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
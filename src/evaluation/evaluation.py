import torch
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.config import LABELS, CNN_MODEL_PATH
from src.training.train_cnn import EmotionClassifierCNN, create_cnn_tensors
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


device = "cuda" if torch.cuda.is_available() else "cpu"
def load_cnn_model():
    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    return model

def confusion_matrix_cnn(): # row normalized confusion matrix for CNN predictions on test set
    model = load_cnn_model()
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = create_cnn_tensors()

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    # normalize by rows
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_normamlized = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_normalized,2), display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('CNN Confusion Matrix — FER2013 Test Set')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(ax.images[0], cax=cax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix_cnn.png', dpi=150)
    plt.show()

    fig1, ax = plt.subplots(figsize=(10, 10))
    disp_normamlized.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('CNN Confusion Matrix — FER2013 Test Set')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(ax.images[0], cax=cax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix_cnn_normalized.png', dpi=150)
    plt.show()

    
# no space in report to plot the grad cam results, but this function is here for reference and can be called in main to visualize the grad cam for a test image
def plot_grad_cam(model, img_tensor, true_label_idx=None):
    """
    img_tensor: shape (1, 1, 48, 48) torch tensor
    """
    # hook to get feature map and gradients from last conv layer
    feature_maps = {}
    gradients = {}

    def forward_hook(module, input, output):
        feature_maps['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # attach hooks to last conv layer
    last_conv = None
    for layer in model.features:
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    last_conv.register_forward_hook(forward_hook)
    last_conv.register_full_backward_hook(backward_hook)

    # forward pass
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_()
    output = model(img_tensor)
    pred_class = torch.argmax(output).item()

    # backward pass on predicted class
    model.zero_grad()
    output[0, pred_class].backward()

    # compute grad-cam
    grads = gradients['value'].squeeze()          # (C, H, W)
    fmaps = feature_maps['value'].squeeze()       # (C, H, W)
    weights = grads.mean(dim=(1, 2))              # (C,)
    cam = torch.zeros(fmaps.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]
    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (48, 48))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # plot
    face = img_tensor.squeeze().detach().cpu().numpy()
    face_uint8 = (face * 255).astype(np.uint8)
    face_rgb = cv2.cvtColor(face_uint8, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(face_rgb, 0.6, heatmap, 0.4, 0)

    pred_label = LABELS[pred_class]
    true_label = LABELS[true_label_idx] if true_label_idx is not None else "unknown"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Grad-CAM | True: {true_label} | Predicted: {pred_label}')
    axes[0].imshow(face, cmap='gray');      axes[0].set_title('Original');  axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet');        axes[1].set_title('Heatmap');   axes[1].axis('off')
    axes[2].imshow(overlay);               axes[2].set_title('Overlay');   axes[2].axis('off')
    plt.tight_layout()
    # plt.savefig(f'gradcam_{pred_label}.png', dpi=150)
    plt.show()




if __name__ == "__main__":
    model = load_cnn_model()
    _, _, X_test_tensor, y_test_tensor = create_cnn_tensors()
    test_dataloader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

    confusion_matrix_cnn()

    plot_grad_cam(model, X_test_tensor[0:1], true_label_idx=y_test_tensor[0].item())
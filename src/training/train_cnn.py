import os
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.preprocess.face_preprocess import setup_files 
from src.config import IMAGE_SIZE, LABELS, OUTPUT_FEATURE_SIZE

class EmotionClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #the convolutional layers that will extract features from the images
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

        #the linears layers that will take the features from the convolutional layers
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, OUTPUT_FEATURE_SIZE),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(OUTPUT_FEATURE_SIZE, num_classes)
        )

    #the forward function that will run the input through the model
    def forward(self, x):
        #run to get features from convulutional layers
        x = self.features(x)

        #run linear layers to get a logits value
        logits = self.linear_relu_stack(x)
        return logits
    

def create_cnn_tensors():
    X_train, y_train, X_test, y_test = setup_files()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def evaluate_cnn(model, test_dataloader, loss_function, device):
    
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            X_batch, y_batch = batch

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            loss = loss_function(output, y_batch)
            
            max_output = torch.max(output, 1)

            eval_loss += loss.item()
            #only gets the predicted labels, not the actual max values
            predicted = max_output.indices

            total += y_batch.size(0)
            for i in range(len(predicted)):
                if predicted[i] == y_batch[i]:
                    correct += 1

        accuracy = correct / total
        average_eval_loss = eval_loss / len(test_dataloader) 
    
    return accuracy, average_eval_loss


def train_cnn(model, device, learning_rate, momentum, num_of_epochs, decay_rate=1e-4):
    from sklearn.model_selection import train_test_split

    #creating tensors and dataloaders for easy training 
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = create_cnn_tensors()

    #creating a validation set from the training data to evaluate the model during training
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_tensor, y_train_tensor, test_size=0.25, random_state=13, stratify=y_train_tensor)

    train_dataset = TensorDataset(X_train_split, y_train_split)
    validation_dataset = TensorDataset(X_val_split, y_val_split)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay_rate)

    epochs_without_improvement = 0
    patience = 3
    best_eval_loss = 10000000 

    for epoch in range(num_of_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            X_batch, y_batch = batch
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            batch_loss = loss_function(output, y_batch)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        average_train_loss = train_loss / len(train_dataloader)
        print("Epoch " + str(epoch) + " of " + str(num_of_epochs) + ", Average Train Loss: " + str(average_train_loss))

        accuracy, current_eval_loss = evaluate_cnn(model, validation_dataloader, loss_function, device)
        print("Epoch " + str(epoch) + " of " + str(num_of_epochs) + ", validation Accuracy: " + str(accuracy) + ", Average validation Loss: " + str(current_eval_loss))

        if current_eval_loss < best_eval_loss - 0.0001:
            best_eval_loss = current_eval_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered after " + str(epoch) + " epochs.")
                break
        

        

    
    accuracy, current_eval_loss = evaluate_cnn(model, test_dataloader, loss_function, device)
    print("Final Test Accuracy: " + str(accuracy) + ", Final Average Test Loss: " + str(current_eval_loss))
    
    # save the model after training
    torch.save(model, "models/cnn_model.pt")
    print("Model saved to models/cnn_model.pt")

def print_model_summary(model):
    print(model)


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model_exists = os.path.exists("models/cnn_model.pt")
    if model_exists:
        print("Model already exists, loading model and continuing training...")
        model = torch.load("models/cnn_model.pt", map_location=device) 
        train_cnn(model, device, learning_rate=0.01, momentum=0.9, num_of_epochs=15, decay_rate=1e-4)
    else:
        print("Model does not exist, creating model and training...")
        model = EmotionClassifierCNN(num_classes=len(LABELS)).to(device)
        train_cnn(model, device, learning_rate=0.01, momentum=0.9, num_of_epochs=15, decay_rate=1e-3)



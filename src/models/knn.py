import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import StratifiedKFold
from src.config import IMAGE_SIZE, LABELS, RAW_TRAIN_DIR
from tqdm import tqdm

training_data = []

#getting a glob to all images for each label classification
for label in LABELS:
    training_data.append(os.path.join(RAW_TRAIN_DIR, label, "*.jpg"))

# might change image_to_vectors to this.    
# will load data from already processed vectors from data/processed
def load_training_data(training_data, labels):
    return 0, 0

# probably change this section to just load already proccessed vectors
#instead of leading the images and proccessing them every time.
# this is redundant now
def image_to_vectors(training_data, labels):
    X_train = []
    Y_train = []

    # reads all jpg's in the training_data folders and turns them into vectors
    for i in range(len(training_data)):
        for img in glob.glob(training_data[i]):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            flatten_image = image.astype(np.float32).flatten() #turn into vector and normalize

            X_train.append(flatten_image)
            Y_train.append(labels[i])

    
    return np.array(X_train), np.array(Y_train)



# does knn prediction for one image.
#calculate the distance from the test image to all training images, 
# find the k closest training images, 
# and return the most common label among those k closest image
def knn_predict_one(X_train, Y_train, X_test, k, labels):
    differences = X_train - X_test
    distances = np.sqrt(np.sum(differences ** 2, axis=1))
    k_closest_indexes = np.argsort(distances)   

    angry = 0
    disgust = 0
    fear = 0
    happy = 0
    sad = 0
    surprise = 0
    neutral = 0

    # calculating the votes 
    for i in k_closest_indexes[0:k]:
        if Y_train[i] == labels[0]:
            angry = angry + 1
        elif Y_train[i] == labels[1]:
            disgust = disgust + 1
        elif Y_train[i] == labels[2]:
            fear = fear + 1
        elif Y_train[i] == labels[3]:
            happy = happy + 1
        elif Y_train[i] == labels[4]:
            sad = sad + 1
        elif Y_train[i] == labels[5]:
            surprise = surprise + 1
        else:
            neutral = neutral + 1
    
    
    counts = [angry, disgust, fear, happy, sad, surprise, neutral]
    max_count = max(angry, disgust, fear, happy, sad, surprise, neutral)

    tied_labels = []
    for i in range(len(counts)):
        if counts[i] == max_count:
            tied_labels.append(i)
    
    if len(tied_labels) == 1:
        return tied_labels[0]

    for i in k_closest_indexes[0:k]:
        if Y_train[i] in tied_labels:
            return Y_train[i]

#runs knn prediction for all test images
def knn_predict(X_train, Y_train, X_test, k, labels):
    preds = []
    for x in tqdm(X_test):
        pred = knn_predict_one(X_train, Y_train, x, k, labels)
        preds.append(pred)

    return np.array(preds)

def accuracy(Y_true, Y_pred):  
    correct = 0

    for i in range(len(Y_true)):
        if Y_true[i] == Y_pred[i]:
            correct = correct + 1

    return correct / len(Y_true)

def f1(Y_true, Y_pred, labels):
    f1_score = 0
    f1_scores = []

    for label in labels:
        TP = 0
        FN = 0
        FP = 0

        for i in range(len(Y_true)):
            if Y_true[i] == label and Y_pred[i] == label:
                TP = TP + 1
            elif Y_true[i] == label and Y_pred[i] != label:
                FN = FN + 1
            elif Y_true[i] != label and Y_pred[i] == label:
                FP = FP + 1

        if(TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        
        if(TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        f1_scores.append(f1_score)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return np.mean(f1_scores)


def cross_validate_knn(X_train, Y_train, k_values, labels, num_folds):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=13)
    fold_accuracies = []
    fold_F1s = []

    for train_index, test_index in skf.split(X_train, Y_train):
        X_train_fold = X_train[train_index]
        Y_train_fold = Y_train[train_index]

        X_test_fold = X_train[test_index]
        Y_test_fold = Y_train[test_index]

        Y_pred_fold = knn_predict(X_train_fold, Y_train_fold, X_test_fold, k_values, labels)

        fold_accuracies.append(accuracy(Y_test_fold, Y_pred_fold))
        fold_F1s.append(f1(Y_test_fold, Y_pred_fold, labels))

    return np.mean(fold_accuracies), np.mean(fold_F1s)

# runs cross validation and outputs the average accuracy and F1 score across all folds
def evaluate_knn(training_data, labels, k, num_folds):
    X_train, Y_train = image_to_vectors(training_data, labels)
    
    from sklearn.model_selection import train_test_split

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=13) 

    accuracy_score, f1_score = cross_validate_knn(X_train, Y_train, k, labels, num_folds)

    print("KNN Classification Results k = " + str(k) + ":\n")
    print("Accuracy: " + str(accuracy_score) + "\n")
    print("F1-score: " + str(f1_score) + "\n")

def predict_image_emotion(image_path, X_train, Y_train, k, labels):
    #1. load the image in with opencv, turn it greyscale and resize it to 48X48
    image_vector = 0

    #2. flatten the image into a vector and normalize it with flaot32 or float64


    #3. use knn_predict_one to predict the emotion of the image and return it 
    prediction = knn_predict_one(X_train, Y_train, image_vector, k, labels)
    return prediction

if __name__ == "__main__":
    evaluate_knn(training_data, LABELS, k=3, num_folds=10)

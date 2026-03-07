import numpy as np
import cv2
import glob
import os
from src.preprocess.face_preprocess import setup_files
from sklearn.model_selection import StratifiedKFold
from src.config import IMAGE_SIZE, LABELS, RAW_TRAIN_DIR
from tqdm import tqdm


def knn_predict_image(X_train, Y_train, X_test, k, labels):
    """
        Return the KNN prediction of a singular image.
        - Calculate the distance from the test image to all training images.
        - Find the k closest training images.
        - Return the most common label among those k closest image.
    """
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


def knn_predict(X_train, Y_train, X_test, k, labels):
    """Run KNN prediction on all test images in set"""
    preds = []
    for x in tqdm(X_test):
        pred = knn_predict_image(X_train, Y_train, x, k, labels)
        preds.append(pred)

    return np.array(preds)


def accuracy(Y_true, Y_pred):  
    """Return accuracy of KNN evaluation"""
    correct = 0

    for i in range(len(Y_true)):
        if Y_true[i] == Y_pred[i]:
            correct = correct + 1

    return correct / len(Y_true)


def f1(Y_true, Y_pred, labels):
    """Return F1-score of KNN evaulation"""
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


def evaluate_knn(X_train, Y_train, labels, k, num_folds):
    """Run cross validation and output the average accuracy and F1-score across all folds."""
    from sklearn.model_selection import train_test_split

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=13) 

    accuracy_score, f1_score = cross_validate_knn(X_train, Y_train, k, labels, num_folds)

    print("KNN Classification Results k = " + str(k) + ":\n")
    print("Accuracy: " + str(accuracy_score) + "\n")
    print("F1-score: " + str(f1_score) + "\n")


if __name__ == "__main__":
    # Load data
    print("\n--- Setup Data ---")
    X_train, y_train, X_test, y_test = setup_files(train_new=False)
    
    # Evaluate KNN model
    print("\n--- Evaluation ---")
    evaluate_knn(X_train, y_train, LABELS, k=3, num_folds=10)

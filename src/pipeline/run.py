import os
import numpy as np
from src.preprocess.face_preprocess import preprocess_image, setup_files, preprocess
from src.config import LABELS
from src.models.knn import knn_predict, accuracy, cross_validate_knn


# Test on a single image 
test_image_path = r"data\raw\\test\angry\PrivateTest_88305.jpg"


if os.path.exists(test_image_path):
    try:
        vector = preprocess_image(test_image_path)
        print(f"Success! Vector shape: {vector.shape}")
        print(f"Vector dtype: {vector.dtype}")
        print(f"Min value: {vector.min()}, Max value: {vector.max()}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Test image not found at {test_image_path}")

X_train, y_train, X_test, y_test = setup_files()
print(f"Loaded training data: {X_train.shape}, {y_train.shape}\nLoaded test data: {X_test.shape}, {y_test.shape}")

# # train data
# train_dir = r"data\raw\train"
# test_dir = r"data\raw\test"
# X_train, y_train = preprocess(train_dir)
# print(f"Preprocessed training data: {X_train.shape}, {y_train.shape}")

# # test data
# X_test, y_test = preprocess(test_dir)
# print(f"Preprocessed test data: {X_test.shape}, {y_test.shape}")

# # Save preprocessed data to .npy files
# np.save("data/processed/train_vectors.npy", X_train)
# np.save("data/processed/train_labels.npy", y_train)
# np.save("data/processed/test_vectors.npy", X_test)
# np.save("data/processed/test_labels.npy", y_test)

# load test data if necessary

# Find best k using cross-validation, cross-validation is slow so sue normal knn to teset
for k in [3,5,7,9]:
    Y_pred = knn_predict(X_train, y_train, X_test, 5, LABELS)
    test_acc = accuracy(y_test, Y_pred)
    print(f"k=5: Test accuracy = {test_acc:.4f}")
    # avg_acc, avg_f1 = cross_validate_knn(X_train, y_train, k, labels=LABELS, num_folds=5)
    # print(f"k={k}: Mean accuracy = {avg_acc:.4f}, Mean F1 = {avg_f1:.4f}")
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import sys

model_filename = "svm_model.pkl"
pca_filename = "pca_instance.pkl"


loaded_model = joblib.load(model_filename)
loaded_pca = joblib.load(pca_filename)

# Read and preprocess the new image as before
new_image_path = sys.argv[1]
new_rgb_image = cv2.imread(new_image_path)
new_rgb_image = cv2.resize(new_rgb_image, (492, 702))
new_gray_image = cv2.cvtColor(new_rgb_image, cv2.COLOR_BGR2GRAY)
new_flattened_image = new_gray_image.flatten()

# Transform the new image using the loaded PCA instance (note: need to reshape to a 2D array)
new_feature_vector = loaded_pca.transform(new_flattened_image.reshape(1, -1))

# Predict the label for the new image using the loaded model
predicted_label = loaded_model.predict(new_feature_vector)

print("Predicted Label:", predicted_label[0])


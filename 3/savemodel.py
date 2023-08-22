import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

folders_path = './seg.91'

# Initialize lists to store features and labels
features = []
labels = []
# Iterate through each folder
for folder_name in os.listdir(folders_path):
    folder_path = os.path.join(folders_path, folder_name)

    # Iterate through each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Read the RGB image and convert it to grayscale
        rgb_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Flatten the image to a 1D vector
        flattened_image = gray_image.flatten()

        # Append the flattened image to the feature list
        features.append(flattened_image)

        # Append the label of the folder to the label list
        labels.append(folder_name)


# Convert the lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Perform PCA
pca = PCA(n_components=100)  # Set the desired number of components
X_pca = pca.fit_transform(X)

# Train an SVM model
svm = SVC(kernel ="linear")
svm.fit(X_pca, y)

model_filename = "svm_model.pkl"
pca_filename = "pca_instance.pkl"
joblib.dump(svm, model_filename)
joblib.dump(pca, pca_filename)



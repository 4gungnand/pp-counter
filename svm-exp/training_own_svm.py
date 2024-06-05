import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Load datasets
people_folder = 'datasets/1'
non_people_folder = 'datasets/0'

people_images, people_labels = load_images_from_folder(people_folder, 1)
non_people_images, non_people_labels = load_images_from_folder(non_people_folder, 0)

X = people_images + non_people_images
y = people_labels + non_people_labels

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}

def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, **hog_params)
        hog_features.append(features)
    return np.array(hog_features)

# Extract HOG features
X_hog = extract_hog_features(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = LinearSVC()
svm.fit(X_train, y_train)

# Save the model
joblib.dump(svm, 'svm_model.pkl')

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy}')

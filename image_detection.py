import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Function to extract HOG features from an image
def extract_hog_features(image):
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    return features

# Function to perform sliding window detection
def sliding_window(image, window_size, stride):
    windows = []
    image_height, image_width = image.shape[:2]
    window_height, window_width = window_size
    for y in range(0, image_height - window_height + 1, stride):
        for x in range(0, image_width - window_width + 1, stride):
            window = image[y:y + window_height, x:x + window_width]
            windows.append(((x, y), (x + window_width, y + window_height), window))
    return windows

# Function to classify windows using the trained SVM classifier
def classify_windows(windows, svm_classifier, scaler):
    classified_windows = []
    for (x1, y1), (x2, y2), window in windows:
        window = cv2.resize(window, (64, 128))
        hog_features = extract_hog_features(window).reshape(1, -1)
        scaled_features = scaler.transform(hog_features)
        prediction = svm_classifier.predict(scaled_features)
        if prediction == 1:
            classified_windows.append(((x1, y1), (x2, y2), prediction))
    return classified_windows

# Load the trained model and scaler
with open('svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example usage on an image
image_path = 'Dataset-CV\Bird View\IMG_2441.jpg'
image = cv2.imread(image_path)
windows = sliding_window(image, window_size=(64, 128), stride=8)
classified_windows = classify_windows(windows, svm_classifier, scaler)

# Draw bounding boxes around classified windows
for (x1, y1), (x2, y2), _ in classified_windows:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Detected People', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
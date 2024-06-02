"The following code is used to detect people in an image using a sliding window approach and a linear SVM classifier."

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

# Function to extract HOG features from an image
def extract_hog_features(image):
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)
    return features

# Function to extract features from a list of images
def extract_features(images):
    features = []
    for image in images:
        hog_features = extract_hog_features(image)
        features.append(hog_features)
    return np.array(features)

# Function to train an SVM classifier
def train_svm_classifier(pos_features, neg_features):
    X = np.vstack((pos_features, neg_features))
    y = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_scaled, y)
    return svm_classifier, scaler

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
    for _, _, window in windows:
        hog_features = extract_hog_features(window)
        scaled_features = scaler.transform([hog_features])
        prediction = svm_classifier.predict(scaled_features)
        if prediction == 1:
            classified_windows.append(window)
    return classified_windows

# Example usage
# Assuming you have a list of positive and negative images
positive_images = [...]  # List of images containing people
negative_images = [...]  # List of images no    t containing people

# Extract HOG features for positive and negative images
positive_features = extract_features(positive_images)
negative_features = extract_features(negative_images)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.vstack((positive_features, negative_features)),
                                                    np.hstack((np.ones(len(positive_features)), np.zeros(len(negative_features)))),
                                                    test_size=0.2, random_state=42)

# Train SVM classifier
svm_classifier, scaler = train_svm_classifier(X_train, y_train)

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
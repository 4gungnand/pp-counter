import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pickle

# Load pre-trained SVM model (You need to train it or use a pre-trained one)
with open('svm_model.pkl', 'rb') as file:
    svm = pickle.load(file)

# Define HOG parameters
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True
}

# Define sliding window parameters
window_size = (64, 128)
step_size = 16

def detect_people(frame):
    people_count = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = gray.shape

    for y in range(0, frame_height - window_size[1], step_size):
        for x in range(0, frame_width - window_size[0], step_size):
            window = gray[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            features = hog(window, **hog_params).reshape(1, -1)
            prediction = svm.predict(features)

            if prediction == 1:
                people_count += 1
                cv2.rectangle(frame, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    return frame, people_count

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, people_count = detect_people(frame)
        cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('People Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'Input\\old_woman_cut.mp4'  # Replace with your video file path
    process_video(video_path)

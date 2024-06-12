import streamlit as st
import cv2
from PIL import Image
import numpy as np
import image_detection

st.title('Object Detection App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert the image to grayscale and resize it
    image = image_detection.preprocess_image(image)

    # Define ROI coordinates as a percentage of the image dimensions
    xmin_pct = 0.1  # 10% from the left
    ymin_pct = 0.3  # 20% from the top
    xmax_pct = 0.9  # 90% to the right
    ymax_pct = 0.8  # 80% to the bottom

    # Calculate the actual ROI coordinates
    height, width = image.shape
    xmin = int(xmin_pct * width)
    ymin = int(ymin_pct * height)
    xmax = int(xmax_pct * width)
    ymax = int(ymax_pct * height)

    # Crop the image using the ROI
    roi = image[ymin:ymax, xmin:xmax]

    # Perform sliding window detection
    windows = image_detection.sliding_window(roi, image_detection.IMAGE_SIZE, stride=8)

    # Classify the windows
    classified_windows = image_detection.classify_windows(windows, image_detection.svm_classifier, image_detection.scaler)

    boxes = np.array([(x1, y1, x2, y2) for (x1, y1), (x2, y2), _ in classified_windows])

    # Apply non-max suppression
    nms = image_detection.non_max_suppression(boxes, 0.3)

    # Draw bounding boxes around classified windows
    for (x1, y1, x2, y2) in nms:
        # Adjust the coordinates based on the ROI
        x1 += xmin
        y1 += ymin
        x2 += xmin
        y2 += ymin
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(image, caption='Processed Image', use_column_width=True)
    st.write("Detected Objects: ", len(nms))
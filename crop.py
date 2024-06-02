import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Directory containing the images and XML files
input_dir = 'dataset/fcv'

# Directory to save the cropped images
output_dir = 'dataset/cropped'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Iterate over each XML file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xml"):
        # Parse the XML file
        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()

        # Read the corresponding image
        img_filename = root.find('filename').text
        img = cv2.imread(os.path.join(input_dir, img_filename))

        # Iterate over each 'object' element in the XML file
        for i, obj in enumerate(root.iter('object')):
            # Get the bounding box coordinates
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)

            # Crop the object from the image
            crop_img = img[y1:y2, x1:x2]

            # Save the cropped image
            cv2.imwrite(os.path.join(output_dir, f'{img_filename.split(".")[0]}_crop_{i}.jpg'), crop_img)
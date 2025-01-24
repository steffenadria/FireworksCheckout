'''
This program uses the previously-trained weighs to make predictions
on the files in the folder "Test Images".
'''

import os
from ultralytics import YOLO

# Class names
CLASSES = [
    'AllBlue', 'Apocalypse', 'Envy', 'GlitteringBrocades',
    'HardCore', 'LoudAndClear', 'MasterBlaster', 'MysticalSky',
    'Phantom', 'RainbowCoconut', 'ShortCircuit', 'Snap',
    'StrobingCoconut', 'ThunderingRainbow'
]

# Load the YOLO model using the training from my final training set
model = YOLO(
    r"C:\Users\Owner\VideosProject\VideoPython\best.pt",
    task="obb"  # Oriented bounding box mode
)

# Folder containing test images
IMAGE_FOLDER = r"C:\Users\Owner\VideosProject\VideoPython"

# Get all image file paths from the folder
image_files = [
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and f.lower().endswith(('.png'))
]

# Run predictions on all images
results = model.predict(
    source=image_files,
    imgsz=1600,  #Use the size of the most accurate training run
    conf=0.6,
)

# Print predictions for each image
for i, result in enumerate(results):
    print(f"\n--- Results for image {os.path.basename(image_files[i])} ---")

    if not result.obb or len(result.obb) == 0:
        print("  No OBB detections for this image.")
    else:
        for obb_box in result.obb:
            cls_idx = int(obb_box.cls)
            confidence = float(obb_box.conf)
            class_name = CLASSES[cls_idx] if 0 <= cls_idx < len(CLASSES) else f"unknown_{cls_idx}"
            print(f"  Class: {class_name}, Confidence: {confidence:.2f}")
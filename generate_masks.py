
import cv2
import numpy as np
import glob
import os

os.makedirs("evaluation/masks", exist_ok=True)

frames = sorted(glob.glob("evaluation/frames/*.jpg"))

for frame_path in frames:
    image = cv2.imread(frame_path)

    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect green background
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    filename = os.path.basename(frame_path).replace(".jpg", ".png")
    output_path = os.path.join("evaluation/masks", filename)

    cv2.imwrite(output_path, mask)

    print(f"Generated: {output_path}")

print("All masks generated!")

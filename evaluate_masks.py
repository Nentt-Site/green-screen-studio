
import cv2
import numpy as np
import os

frames_dir = "evaluation/frames"
masks_dir = "evaluation/masks"

key_hsv = (48, 240, 248)
h_tol, s_tol, v_tol = 10, 80, 80

accuracies = []

for i in range(1, 6):
    frame_path = f"{frames_dir}/frame{i}.jpg"
    reference_path = f"{masks_dir}/frame{i}.png"

    frame = cv2.imread(frame_path)
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    if frame is None or reference is None:
        print(f"Frame {i}: Missing image or reference mask.")
        continue

    # Generate mask using the application's HSV method
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = key_hsv

    lower = np.array([
        max(0, h - h_tol),
        max(0, s - s_tol),
        max(0, v - v_tol)
    ])

    upper = np.array([
        min(179, h + h_tol),
        min(255, s + s_tol),
        min(255, v + v_tol)
    ])

    generated = cv2.inRange(hsv, lower, upper)

    # Apply the same morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    generated = cv2.morphologyEx(
        generated, cv2.MORPH_OPEN, kernel
    )
    generated = cv2.morphologyEx(
        generated, cv2.MORPH_CLOSE, kernel
    )

    if generated.shape != reference.shape:
        print(f"Frame {i}: Dimension mismatch.")
        continue

    # Compare every pixel
    correct = np.count_nonzero(generated == reference)
    total = generated.size
    accuracy = (correct / total) * 100

    accuracies.append(accuracy)

    print(f"Frame {i} Accuracy: {accuracy:.2f}%")

if accuracies:
    print("\n-------------------------------")
    print(f"Average Pixel Accuracy: {np.mean(accuracies):.2f}%")
    print("-------------------------------")

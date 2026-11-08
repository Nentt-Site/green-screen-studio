
import cv2
import numpy as np
import os

frames_dir = "evaluation/frames"
masks_dir = "evaluation/masks"

frames = [f"frame{i}.jpg" for i in range(1, 6)]
index = 0
brush_size = 10
mask = None

def save_mask():
    path = os.path.join(
        masks_dir, frames[index].replace(".jpg", ".png")
    )
    cv2.imwrite(path, mask)
    print(f"Saved {path}")

def mouse(event, x, y, flags, param):
    global mask

    if event == cv2.EVENT_LBUTTONDOWN or (
        event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
    ):
        cv2.circle(mask, (x * 2, y * 2), brush_size * 2, 0, -1)

    elif event == cv2.EVENT_RBUTTONDOWN or (
        event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON
    ):
        cv2.circle(mask, (x * 2, y * 2), brush_size * 2, 255, -1)

def load_frame():
    global mask

    frame = cv2.imread(os.path.join(frames_dir, frames[index]))
    mask = cv2.imread(
        os.path.join(masks_dir, frames[index].replace(".jpg", ".png")),
        0
    )

    cv2.imshow("Original Frame", cv2.resize(frame, (960, 540)))
    cv2.imshow("Reference Mask", cv2.resize(mask, (960, 540)))

cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Reference Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Frame", 960, 540)
cv2.resizeWindow("Reference Mask", 960, 540)
cv2.setMouseCallback("Reference Mask", mouse)

load_frame()

print("""
LEFT CLICK: Mark person black
RIGHT CLICK: Mark background white
+ / -: Change brush size
N: Next frame
P: Previous frame
S: Save current mask
Q: Quit
""")

while True:
    cv2.imshow(
        "Reference Mask",
        cv2.resize(mask, (960, 540))
    )

    key = cv2.waitKey(30) & 0xFF

    if key == ord('n'):
        save_mask()
        index = min(index + 1, 4)
        load_frame()

    elif key == ord('p'):
        save_mask()
        index = max(index - 1, 0)
        load_frame()

    elif key == ord('s'):
        save_mask()

    elif key in (ord('+'), ord('=')):
        brush_size = min(brush_size + 2, 50)

    elif key in (ord('-'), ord('_')):
        brush_size = max(brush_size - 2, 2)

    elif key == ord('q'):
        save_mask()
        break

cv2.destroyAllWindows()

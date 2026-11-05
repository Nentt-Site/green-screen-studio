
import cv2
import numpy as np


# ==========================================
# FILTER FUNCTIONS
# ==========================================

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_brightness(image, value=50):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


def apply_contrast(image, value=1.5):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)


def apply_sepia(image):
    # Transformation matrix for BGR input and output
    matrix = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393]
    ])

    return cv2.transform(image, matrix).clip(0, 255).astype(np.uint8)


# ==========================================
# AVAILABLE FILTERS
# ==========================================

FILTERS = [
    "normal",
    "grayscale",
    "brightness",
    "darkness",
    "high_contrast",
    "low_contrast",
    "sepia"
]


def apply_filter(image, filter_name="normal"):

    if filter_name == "grayscale":
        return apply_grayscale(image)

    elif filter_name == "brightness":
        return apply_brightness(image, 50)

    elif filter_name == "darkness":
        return apply_brightness(image, -50)

    elif filter_name == "high_contrast":
        return apply_contrast(image, 1.8)

    elif filter_name == "low_contrast":
        return apply_contrast(image, 0.5)

    elif filter_name == "sepia":
        return apply_sepia(image)

    elif filter_name == "normal":
        return image.copy()

    else:
        raise ValueError(f"Unknown filter: {filter_name}")


# ==========================================
# GREEN SCREEN + BACKGROUND REPLACEMENT
# ==========================================

def remove_green_screen(frame, background):

    if frame is None or background is None:
        raise ValueError("Frame and background must be valid images.")

    # Resize replacement background
    bg = cv2.resize(
        background,
        (frame.shape[1], frame.shape[0])
    )

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green colour range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Detect green pixels
    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Extract subject
    subject_mask = cv2.bitwise_not(mask)

    person = cv2.bitwise_and(
        frame,
        frame,
        mask=subject_mask
    )

    # Extract replacement background
    new_background = cv2.bitwise_and(
        bg,
        bg,
        mask=mask
    )

    # Combine both images
    return cv2.add(person, new_background)


# ==========================================
# MAIN INTEGRATION FUNCTION
# ==========================================

def process_frame(frame, background, filter_name="normal"):

    """
    Processes one webcam frame.

    Parameters:
        frame: Current webcam frame (BGR image).
        background: Selected replacement background (BGR image).
        filter_name: Selected filter.

    Returns:
        Processed BGR image.
    """

    result = remove_green_screen(frame, background)

    output = apply_filter(result, filter_name)

    return output

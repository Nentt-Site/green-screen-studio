
import cv2
import numpy as np


def create_hsv_mask(frame, key_hsv, h_tol=10, s_tol=80, v_tol=80):
    """Detect the selected colour using HSV distance thresholds."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = [int(x) for x in key_hsv]

    lower = np.array([
        max(0, s - s_tol),
        max(0, v - v_tol)
    ])

    upper = np.array([
        min(255, s + s_tol),
        min(255, v + v_tol)
    ])

    # Handle hue wrap-around at 0 and 179
    low_h = h - h_tol
    high_h = h + h_tol

    if low_h < 0:
        mask1 = cv2.inRange(
            hsv,
            np.array([0, lower[0], lower[1]]),
            np.array([high_h, upper[0], upper[1]])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([180 + low_h, lower[0], lower[1]]),
            np.array([179, upper[0], upper[1]])
        )
        mask = cv2.bitwise_or(mask1, mask2)

    elif high_h > 179:
        mask1 = cv2.inRange(
            hsv,
            np.array([low_h, lower[0], lower[1]]),
            np.array([179, upper[0], upper[1]])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([0, lower[0], lower[1]]),
            np.array([high_h - 180, upper[0], upper[1]])
        )
        mask = cv2.bitwise_or(mask1, mask2)

    else:
        mask = cv2.inRange(
            hsv,
            np.array([low_h, lower[0], lower[1]]),
            np.array([high_h, upper[0], upper[1]])
        )

    return mask


def create_rgb_mask(frame, key_rgb, tolerance=60):
    """Detect the selected colour using Euclidean RGB distance."""

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    key = np.array(key_rgb, dtype=np.float32)

    distance = np.sqrt(np.sum((rgb - key) ** 2, axis=2))

    mask = np.where(distance <= tolerance, 255, 0)

    return mask.astype(np.uint8)


def clean_mask(mask, kernel_size=5):
    """Remove small noise and fill small holes."""

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    # Opening removes small foreground noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Closing fills small holes
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def create_soft_alpha(mask, feather=5):
    """Create a smooth background alpha mask."""

    if feather <= 0:
        return mask.astype(np.float32) / 255.0

    kernel_size = feather * 2 + 1

    soft_mask = cv2.GaussianBlur(
        mask,
        (kernel_size, kernel_size),
        0
    )

    return soft_mask.astype(np.float32) / 255.0


def suppress_green_spill(frame, mask):
    """Reduce green contamination along the subject's edges."""

    result = frame.copy()

    kernel = np.ones((3, 3), np.uint8)

    expanded = cv2.dilate(mask, kernel, iterations=1)

    # Subject pixels immediately beside the detected background
    edge = cv2.bitwise_and(
        expanded,
        cv2.bitwise_not(mask)
    )

    b, g, r = cv2.split(result)

    # Reduce excessive green on edge pixels
    corrected_green = np.minimum(
        g.astype(np.float32),
        np.maximum(r, b).astype(np.float32) * 1.1
    )

    g[edge > 0] = corrected_green[edge > 0].astype(np.uint8)

    return cv2.merge((b, g, r))


def composite_background(frame, background, mask, feather=5):
    """Combine the foreground with a replacement background."""

    height, width = frame.shape[:2]

    background = cv2.resize(background, (width, height))

    alpha = create_soft_alpha(mask, feather)

    # Background alpha: 1 means replacement background
    alpha = alpha[:, :, np.newaxis]

    result = (
        frame.astype(np.float32) * (1 - alpha)
        + background.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def process_chroma_key(
    frame,
    background,
    key_hsv,
    h_tol=10,
    s_tol=80,
    v_tol=80,
    feather=5,
    use_rgb=False,
    rgb_tolerance=60,
    clean=True,
    spill=True
):
    """Complete chroma-key processing pipeline."""

    if frame is None or background is None:
        raise ValueError("Frame and background are required.")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    key_rgb = cv2.cvtColor(
        np.uint8([[key_hsv]]),
        cv2.COLOR_HSV2RGB
    )[0, 0]

    if use_rgb:
        mask = create_rgb_mask(
            frame,
            key_rgb,
            rgb_tolerance
        )
    else:
        mask = create_hsv_mask(
            frame,
            key_hsv,
            h_tol,
            s_tol,
            v_tol
        )

    if clean:
        mask = clean_mask(mask)

    foreground = suppress_green_spill(frame, mask) if spill else frame

    output = composite_background(
        foreground,
        background,
        mask,
        feather
    )

    return output, mask

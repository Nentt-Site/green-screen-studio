
import cv2
import numpy as np
import os

# ==========================================
# 1. FILTER FUNCTIONS
# ==========================================

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


def apply_contrast(image, value):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)


def apply_sepia(image):
    matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    return cv2.transform(image, matrix).clip(0, 255).astype(np.uint8)


# ==========================================
# 2. LOAD BACKGROUND AND WEBCAM
# ==========================================

background = cv2.imread("green_background.jpeg")

if background is None:
    print("Error: Background image not found!")
    exit()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access webcam!")
    exit()


# ==========================================
# 3. INITIAL SETTINGS
# ==========================================

current_filter = "normal"

os.makedirs("output", exist_ok=True)

writer = None
recording = False

print("""
=================================
       GREEN SCREEN STUDIO
=================================

FILTER CONTROLS

1 = Normal
2 = Grayscale
3 = Brightness
4 = Darkness
5 = High Contrast
6 = Low Contrast
7 = Sepia

RECORDING CONTROLS

R = Start Recording
S = Stop Recording
Q = Quit

=================================
""")


# ==========================================
# 4. LIVE VIDEO PROCESSING
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        # Mirror webcam preview
        frame = cv2.flip(frame, 1)

        # Resize background to match webcam
        bg = cv2.resize(
            background,
            (frame.shape[1], frame.shape[0])
        )

        # ==================================
        # 5. GREEN SCREEN REMOVAL
        # ==================================

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        subject_mask = cv2.bitwise_not(mask)

        # Extract person
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

        # Combine both
        result = cv2.add(person, new_background)

        # ==================================
        # 6. APPLY SELECTED FILTER
        # ==================================

        if current_filter == "grayscale":
            output = apply_grayscale(result)

        elif current_filter == "brightness":
            output = apply_brightness(result, 50)

        elif current_filter == "darkness":
            output = apply_brightness(result, -50)

        elif current_filter == "high":
            output = apply_contrast(result, 1.8)

        elif current_filter == "low":
            output = apply_contrast(result, 0.5)

        elif current_filter == "sepia":
            output = apply_sepia(result)

        else:
            output = result.copy()

        # ==================================
        # 7. RECORDING STATUS
        # ==================================

        if recording:

            cv2.putText(
                output,
                "RECORDING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            writer.write(output)

        # Display final output
        cv2.imshow(
            "Green Screen Studio",
            output
        )

        # ==================================
        # 8. KEYBOARD CONTROLS
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        # Filter selection
        if key == ord('1'):
            current_filter = "normal"

        elif key == ord('2'):
            current_filter = "grayscale"

        elif key == ord('3'):
            current_filter = "brightness"

        elif key == ord('4'):
            current_filter = "darkness"

        elif key == ord('5'):
            current_filter = "high"

        elif key == ord('6'):
            current_filter = "low"

        elif key == ord('7'):
            current_filter = "sepia"

        # Start recording
        elif key == ord('r') and not recording:

            height, width = output.shape[:2]

            writer = cv2.VideoWriter(
                "output/recording.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                20.0,
                (width, height)
            )

            if writer.isOpened():
                recording = True
                print("Recording started!")

            else:
                print("Could not start recording!")
                writer.release()
                writer = None

        # Stop recording
        elif key == ord('s') and recording:

            recording = False
            writer.release()
            writer = None

            print("Recording saved to output/recording.mp4")

        # Quit
        elif key == ord('q'):
            break

finally:

    # Release recording resources
    if writer is not None:
        writer.release()

    # Release webcam
    camera.release()

    # Close windows
    cv2.destroyAllWindows()

    print("Green Screen Studio closed.")


import cv2

from modules.image_processing import process_frame, FILTERS
from modules.video_recorder import VideoRecorder


# ==========================================
# LOAD BACKGROUND
# ==========================================

background = cv2.imread(
    "assets/backgrounds/background.jpeg"
)

if background is None:
    raise FileNotFoundError("Replacement background not found.")


# ==========================================
# OPEN WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")


# ==========================================
# INITIAL SETTINGS
# ==========================================

current_filter = "normal"

recorder = VideoRecorder()

print("""
GREEN SCREEN STUDIO DEMO

FILTERS
1 = Normal
2 = Grayscale
3 = Brightness
4 = Darkness
5 = High Contrast
6 = Low Contrast
7 = Sepia

RECORDING
R = Start recording
S = Stop recording
Q = Quit
""")


filter_keys = {
    ord("1"): "normal",
    ord("2"): "grayscale",
    ord("3"): "brightness",
    ord("4"): "darkness",
    ord("5"): "high_contrast",
    ord("6"): "low_contrast",
    ord("7"): "sepia"
}


# ==========================================
# LIVE PROCESSING
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Process original orientation
        output = process_frame(
            frame,
            background,
            current_filter
        )

        # Record unmirrored processed output
        if recorder.is_recording:
            recorder.write(output)

        # Mirror only the preview
        preview = cv2.flip(output, 1)

        if recorder.is_recording:
            cv2.putText(
                preview,
                "RECORDING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Green Screen Studio", preview)

        key = cv2.waitKey(1) & 0xFF

        # Select filter
        if key in filter_keys:
            current_filter = filter_keys[key]
            print("Selected filter:", current_filter)

        # Start recording
        elif key == ord("r"):

            if not recorder.is_recording:
                try:
                    recorder.start(output)
                    print("Recording started!")
                except RuntimeError as error:
                    print(error)

        # Stop recording
        elif key == ord("s"):

            saved_path = recorder.stop()

            if saved_path:
                print("Recording saved:", saved_path)

        # Quit
        elif key == ord("q"):
            break

finally:

    if recorder.is_recording:
        print("Recording saved:", recorder.stop())

    camera.release()
    cv2.destroyAllWindows()

    print("Green Screen Studio demo closed.")

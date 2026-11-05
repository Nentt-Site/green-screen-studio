
import cv2
import os


class VideoRecorder:

    def __init__(self, output_dir="output", fps=20):
        self.output_dir = output_dir
        self.fps = fps
        self.writer = None
        self.filepath = None
        self.frame_size = None

        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def is_recording(self):
        return self.writer is not None

    def start(self, frame, filename="recording.mp4"):

        if frame is None:
            raise ValueError("Cannot record an empty frame.")

        if self.is_recording:
            return False

        height, width = frame.shape[:2]
        self.frame_size = (width, height)

        self.filepath = os.path.join(
            self.output_dir,
            filename
        )

        self.writer = cv2.VideoWriter(
            self.filepath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.frame_size
        )

        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            self.frame_size = None
            raise RuntimeError("Could not start video recording.")

        return True

    def write(self, frame):

        if not self.is_recording or frame is None:
            return False

        height, width = frame.shape[:2]

        if (width, height) != self.frame_size:
            raise ValueError("Frame dimensions changed during recording.")

        self.writer.write(frame)
        return True

    def stop(self):

        if not self.is_recording:
            return None

        self.writer.release()
        self.writer = None

        saved_path = self.filepath

        self.filepath = None
        self.frame_size = None

        return saved_path

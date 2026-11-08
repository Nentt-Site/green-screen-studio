import customtkinter as ctk
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

from modules.chroma_key import (
    process_chroma_key, create_hsv_mask, create_rgb_mask, clean_mask
)
from modules.image_processing import apply_filter
from modules.video_recorder import VideoRecorder


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Green Screen Studio")
app.geometry("1320x850")
app.minsize(1120, 720)

PREVIEW_W, PREVIEW_H = 700, 420
camera = None
camera_after_id = None
background = cv2.imread("assets/backgrounds/background.jpg")
background_video = None
latest_output = None
latest_frame = None
key_hsv = (60, 255, 255)
key_rgb = (0, 255, 0)
selected_filter = "normal"
use_rgb = False
h_tolerance, s_tolerance, v_tolerance = 10, 80, 80
rgb_tolerance, feather = 60, 5
recorder = VideoRecorder()
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

FILTER_OPTIONS = {
    "Normal": "normal", "Grayscale": "grayscale",
    "Brightness": "brightness", "Darkness": "darkness",
    "High Contrast": "high_contrast", "Low Contrast": "low_contrast",
    "Sepia": "sepia", "Invert": "invert",
    "Posterize": "posterize", "CMYK Separation": "cmyk"
}


def show_image(label, frame, size):
    if frame is None:
        return
    display = cv2.resize(frame, size)
    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    label.configure(image=photo, text="")
    label.image = photo


def change_filter(choice):
    global selected_filter
    selected_filter = FILTER_OPTIONS[choice]
    status_label.configure(text=f"Filter: {choice}")


def choose_background():
    global background, background_video
    path = filedialog.askopenfilename(
        title="Choose replacement background",
        filetypes=[
            ("Images and videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv")
        ]
    )
    if not path:
        return

    if background_video is not None:
        background_video.release()
        background_video = None

    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Background error", "Could not open that video.")
            return
        background = None
        background_video = cap
        status_label.configure(text="Video background loaded")
    else:
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Background error", "Could not load that image.")
            return
        background = image
        status_label.configure(text="Image background loaded")


def next_background_frame():
    global background_video
    if background_video is None:
        return background
    ok, frame = background_video.read()
    if not ok:
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = background_video.read()
    return frame if ok else None


def pick_key_color(event):
    global key_hsv, key_rgb
    if latest_frame is None:
        return

    height, width = latest_frame.shape[:2]
    shown_w = max(1, preview.winfo_width())
    shown_h = max(1, preview.winfo_height())
    x = width - 1 - int(event.x * width / shown_w)  # preview is mirrored
    y = int(event.y * height / shown_h)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    pixel = latest_frame[y, x]
    key_hsv = tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[pixel]]), cv2.COLOR_BGR2HSV
    )[0, 0])
    key_rgb = tuple(int(v) for v in cv2.cvtColor(
        np.uint8([[pixel]]), cv2.COLOR_BGR2RGB
    )[0, 0])
    status_label.configure(text=f"Key colour selected — HSV {key_hsv}, RGB {key_rgb}")


def update_h(value):
    global h_tolerance
    h_tolerance = int(value)
    h_value.configure(text=f"H tolerance: {h_tolerance}")


def update_s(value):
    global s_tolerance
    s_tolerance = int(value)
    s_value.configure(text=f"S tolerance: {s_tolerance}")


def update_v(value):
    global v_tolerance
    v_tolerance = int(value)
    v_value.configure(text=f"V tolerance: {v_tolerance}")


def update_rgb(value):
    global rgb_tolerance
    rgb_tolerance = int(value)
    rgb_value.configure(text=f"RGB distance: {rgb_tolerance}")


def update_feather(value):
    global feather
    feather = int(value)
    feather_value.configure(text=f"Edge feather: {feather}")


def choose_mask_mode():
    global use_rgb
    use_rgb = mask_mode.get() == "RGB"
    status_label.configure(text=f"Keying mode: {'RGB distance' if use_rgb else 'HSV'}")


def make_histogram(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    selected = hue[mask > 0]
    canvas = np.zeros((130, 300, 3), dtype=np.uint8)
    if selected.size:
        hist = np.bincount(selected, minlength=180).astype(np.float32)
        hist /= max(1.0, hist.max())
        points = []
        for x in range(180):
            points.append((int(x * 299 / 179), 124 - int(hist[x] * 115)))
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (80, 220, 120), 2)
    cv2.putText(canvas, "Live hue histogram (key region)", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
    return canvas




def update_camera():
    global latest_output, latest_frame, camera_after_id
    global fps_start_time, fps_frame_count, current_fps

    if camera is None:
        return

    ok, frame = camera.read()
    if ok:
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time

        if elapsed >= 1:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        latest_frame = frame.copy()
        bg = next_background_frame()

        hsv_mask = create_hsv_mask(frame, key_hsv, h_tolerance, s_tolerance, v_tolerance)
        rgb_mask = create_rgb_mask(frame, key_rgb, rgb_tolerance)
        selected_mask = rgb_mask if use_rgb else hsv_mask
        selected_mask = clean_mask(selected_mask)

        if bg is not None:
            output, _, hsv_mask, rgb_mask = process_chroma_key(
                frame, bg, key_hsv, key_rgb,
                h_tol=h_tolerance, s_tol=s_tolerance, v_tol=v_tolerance,
                feather=feather, rgb_tolerance=rgb_tolerance,
                clean=True, spill=True, use_rgb=use_rgb
            )
        else:
            output = frame.copy()

        output = apply_filter(output, selected_filter)
        latest_output = output

        if recorder.is_recording:
            record_frame = output
            if selected_filter == "cmyk":
                record_frame = frame
            recorder.write(record_frame)

        # Display FPS on the camera preview
        fps_display = cv2.flip(output, 1).copy()
        cv2.putText(
            fps_display,
            f"FPS: {current_fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        show_image(preview, fps_display, (PREVIEW_W, PREVIEW_H))

        comparison = np.hstack((
            cv2.cvtColor(cv2.resize(hsv_mask, (150, 95)), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(cv2.resize(rgb_mask, (150, 95)), cv2.COLOR_GRAY2BGR)
        ))
        cv2.putText(comparison, "HSV", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(comparison, "RGB", (158, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        show_image(mask_preview, comparison, (300, 95))
        show_image(histogram_preview, make_histogram(frame, selected_mask), (300, 130))

    camera_after_id = app.after(15, update_camera)


def start_camera():
    global camera
    if camera is not None:
        return
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        camera = None
        messagebox.showerror("Camera error", "Could not access webcam.")
        return
    camera.set(cv2.CAP_PROP_FPS, 30)
    status_label.configure(text="Camera running")
    update_camera()


def stop_camera():
    global camera, camera_after_id
    if camera_after_id is not None:
        app.after_cancel(camera_after_id)
        camera_after_id = None
    if camera is not None:
        camera.release()
        camera = None
    if recorder.is_recording:
        recorder.stop()
    status_label.configure(text="Camera stopped")


def start_recording():
    if camera is None or latest_output is None:
        messagebox.showwarning("Warning", "Start the camera first.")
        return
    if recorder.is_recording:
        return
    try:
        recorder.start(latest_output)
        status_label.configure(text="Recording...")
    except Exception as exc:
        messagebox.showerror("Recording error", str(exc))


def stop_recording():
    if not recorder.is_recording:
        return
    path = recorder.stop()
    status_label.configure(text=f"Recording saved: {path}")
    messagebox.showinfo("Recording complete", f"Saved to:\n{path}")


def on_close():
    stop_camera()
    if background_video is not None:
        background_video.release()
    if recorder.is_recording:
        recorder.stop()
    app.destroy()


# -------------------- Interface --------------------

title = ctk.CTkLabel(app, text="GREEN SCREEN STUDIO", font=("Arial", 24, "bold"))
title.pack(pady=12)

main = ctk.CTkFrame(app)
main.pack(fill="both", expand=True, padx=14, pady=8)

left = ctk.CTkFrame(main)
left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

preview = ctk.CTkLabel(left, text="WEBCAM PREVIEW", width=PREVIEW_W, height=PREVIEW_H)
preview.pack(padx=8, pady=8)
preview.bind("<Button-1>", pick_key_color)

diagnostics = ctk.CTkFrame(left)
diagnostics.pack(fill="x", padx=8, pady=6)

mask_preview = ctk.CTkLabel(diagnostics, text="HSV mask | RGB mask", width=300, height=95)
mask_preview.pack(side="left", padx=5, pady=5)

histogram_preview = ctk.CTkLabel(diagnostics, text="Hue histogram", width=300, height=130)
histogram_preview.pack(side="left", padx=5, pady=5)

controls = ctk.CTkScrollableFrame(main, width=330)
controls.pack(side="right", fill="y", padx=10, pady=10)

ctk.CTkLabel(controls, text="FILTER").pack(pady=(8, 2))
filter_menu = ctk.CTkOptionMenu(controls, values=list(FILTER_OPTIONS.keys()), command=change_filter)
filter_menu.pack(pady=5)

ctk.CTkButton(controls, text="Choose Image / Video Background", command=choose_background).pack(pady=8)
ctk.CTkButton(controls, text="Start Camera", command=start_camera).pack(pady=6)
ctk.CTkButton(controls, text="Stop Camera", command=stop_camera).pack(pady=6)
ctk.CTkButton(controls, text="Start Recording", command=start_recording).pack(pady=6)
ctk.CTkButton(controls, text="Stop Recording", command=stop_recording).pack(pady=6)

ctk.CTkLabel(controls, text="KEY COLOUR").pack(pady=(14, 2))
ctk.CTkLabel(controls, text="Click the green area in the preview to pick it.").pack(pady=3)

mask_mode = ctk.StringVar(value="HSV")
ctk.CTkSegmentedButton(controls, values=["HSV", "RGB"], variable=mask_mode,
                       command=lambda _: choose_mask_mode()).pack(pady=6)

h_value = ctk.CTkLabel(controls, text=f"H tolerance: {h_tolerance}")
h_value.pack()
ctk.CTkSlider(controls, from_=0, to=90, command=update_h).pack(fill="x", padx=12, pady=3)

s_value = ctk.CTkLabel(controls, text=f"S tolerance: {s_tolerance}")
s_value.pack()
ctk.CTkSlider(controls, from_=0, to=255, command=update_s).pack(fill="x", padx=12, pady=3)

v_value = ctk.CTkLabel(controls, text=f"V tolerance: {v_tolerance}")
v_value.pack()
ctk.CTkSlider(controls, from_=0, to=255, command=update_v).pack(fill="x", padx=12, pady=3)

rgb_value = ctk.CTkLabel(controls, text=f"RGB distance: {rgb_tolerance}")
rgb_value.pack()
ctk.CTkSlider(controls, from_=0, to=441, command=update_rgb).pack(fill="x", padx=12, pady=3)

feather_value = ctk.CTkLabel(controls, text=f"Edge feather: {feather}")
feather_value.pack()
ctk.CTkSlider(controls, from_=0, to=20, command=update_feather).pack(fill="x", padx=12, pady=3)

status_label = ctk.CTkLabel(app, text="Ready")
status_label.pack(pady=7)

app.protocol("WM_DELETE_WINDOW", on_close)
app.mainloop()
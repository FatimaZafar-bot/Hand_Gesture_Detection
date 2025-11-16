import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model


MODEL_PATH = r"C:\Users\Hp\Desktop\HandGesture\hand_gesture_model.h5"
LABELS_PATH = r"C:\Users\Hp\Desktop\HandGesture\gesture_labels.json"

model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

class HandGestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hand Gesture Recognition")
        self.window.geometry("800x600")
        self.running = False

        self.video_label = Label(window)
        self.video_label.pack()

        # Gesture label
        self.gesture_label = Label(window, text="Gesture: None", font=("Helvetica", 20))
        self.gesture_label.pack(pady=20)

        self.start_btn = Button(window, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=20)

        self.stop_btn = Button(window, text="Stop", command=self.stop)
        self.stop_btn.pack(side="right", padx=20)

        # Webcam
        self.cap = cv2.VideoCapture(0)

    def start(self):
        if not self.running:
            self.running = True
            self.update_frame()

    def stop(self):
        self.running = False
        self.cap.release()
        self.video_label.config(image='')

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Preprocess frame for CNN
            img = cv2.resize(frame, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img, axis=0) / 255.0  # normalize

            # Predict gesture
            pred = model.predict(img_input, verbose=0)
            class_id = np.argmax(pred)
            gesture_name = inv_class_indices[class_id]
            self.gesture_label.config(text=f"Gesture: {gesture_name}")

            # Display video in Tkinter
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        if self.running:
            self.window.after(30, self.update_frame)  # ~33 FPS

# === Run app ===
if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()

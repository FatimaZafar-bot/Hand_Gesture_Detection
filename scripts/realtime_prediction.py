import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json

# === PATHS ===
MODEL_PATH = r"C:\Users\Hp\Desktop\HandGesture\hand_gesture_model.h5"
LABELS_PATH = r"C:\Users\Hp\Desktop\HandGesture\gesture_labels.json"

# === LOAD MODEL AND LABELS ===
model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

# === MEDIAPIPE HANDS ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === START WEBCAM ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 10
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 10
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 10
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 10

            # Crop and preprocess hand image
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict gesture
            pred = model.predict(hand_img)
            class_id = np.argmax(pred)
            gesture_name = inv_class_indices[class_id]

            # Display gesture on frame
            cv2.putText(frame, gesture_name, (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

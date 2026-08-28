import os
import cv2
import time
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "gesture_model.joblib")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Trained model not found. Run train_model.py first.")

model = joblib.load(MODEL_FILE)

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)

def extract_features(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return [features]

cap = cv2.VideoCapture(0)

# Word Assembly State
current_word = ""
last_prediction = None
stable_frame_count = 0
CONFIRMATION_FRAMES = 20  # ~0.7 seconds at 30 FPS

print("\n--- Word Speller Controls ---")
print("• Hold a gesture steady to type that letter")
print("• Press [Spacebar] on keyboard: Add space")
print("• Press [Backspace]: Delete last letter")
print("• Press [c]: Clear word")
print("• Press [q]: Quit\n")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        detected_char = None
        confidence = 0.0

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            features = extract_features(landmarks)

            # Predict character and probability
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            confidence = max(probs)

            if confidence > 0.65:
                detected_char = pred

            # Draw hand skeleton
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

            # Debounce / Word builder logic
            if detected_char == last_prediction and detected_char is not None:
                stable_frame_count += 1
                if stable_frame_count == CONFIRMATION_FRAMES:
                    current_word += detected_char
            else:
                last_prediction = detected_char
                stable_frame_count = 0

        # HUD & Dynamic Word Box
        progress = min(stable_frame_count / CONFIRMATION_FRAMES, 1.0)
        bar_w = int(progress * 200)
        cv2.rectangle(frame, (20, 45), (20 + bar_w, 55), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (20, 45), (220, 55), (255, 255, 255), 1)

        char_text = f"Live Char: {detected_char} ({int(confidence * 100)}%)" if detected_char else "Live Char: None"
        cv2.putText(frame, char_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display Word Output
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 15), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, f"Spelled Word: {current_word}", (30, h - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Sign Language Word Speller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Space
            current_word += " "
        elif key == 8:   # Backspace
            current_word = current_word[:-1]
        elif key == ord('c'):
            current_word = ""

cap.release()
cv2.destroyAllWindows()
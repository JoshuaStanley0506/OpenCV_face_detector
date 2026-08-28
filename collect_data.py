import os
import cv2
import time
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "gesture_data.csv")

# Initialize MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path=os.path.join(BASE_DIR, "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)

def extract_features(landmarks):
    """Normalizes 21 landmarks relative to the wrist (landmark 0)."""
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return features

cap = cv2.VideoCapture(0)
current_label = None
samples_collected = {}

print("\n--- Data Collection Instructions ---")
print("1. Press any letter key (e.g., 'a', 'b', 'c', 'l', 'v') to select that class.")
print("2. Hold your hand in that gesture and press 'SPACE' to record samples.")
print("3. Press 'q' when finished.\n")

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

        current_features = None
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            current_features = extract_features(landmarks)

            # Draw simple skeleton points
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # UI Overlay
        cv2.putText(frame, f"Active Class: {current_label if current_label else 'None (Press a key)'}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        count = samples_collected.get(current_label, 0)
        cv2.putText(frame, f"Samples for '{current_label}': {count}", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Spacebar to capture
            if current_label and current_features:
                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + current_features)
                samples_collected[current_label] = samples_collected.get(current_label, 0) + 1
                print(f"[+] Recorded sample #{samples_collected[current_label]} for '{current_label}'")
        elif 97 <= key <= 122 or 48 <= key <= 57:  # a-z or 0-9
            current_label = chr(key).upper()
            if current_label not in samples_collected:
                samples_collected[current_label] = 0

cap.release()
cv2.destroyAllWindows()

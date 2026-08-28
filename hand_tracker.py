import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]

TIP_IDS = [8, 12, 16, 20]

def count_fingers(landmarks, handedness_label):
    """Counts extended fingers based on joint Y/X relationships."""
    fingers = []

    # 1. Thumb: check X coordinate relative to IP joint depending on handedness
    if handedness_label == 'Right':
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    # 2. 4 Fingers: tip Y must be higher (lower screen coordinate) than PIP joint
    for tip_id in TIP_IDS:
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return fingers


base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
prev_frame_time = 0

print("\n--- Hand Tracking Controls ---")
print("• Pinch Thumb + Index: Adjust dynamic pinch gauge")
print("• Hold up fingers: Real-time finger counter")
print("• Show Peace / Fist / Open Hand: Live gesture classifier")
print("• Press 'q' to Quit\n")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB and wrap into mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonically increasing timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
        prev_frame_time = current_time

        total_fingers = 0

        if results.hand_landmarks:
            for idx, landmarks in enumerate(results.hand_landmarks):
                handedness_label = results.handedness[idx][0].category_name

                # 1. Draw Skeleton Mesh
                for p1, p2 in HAND_CONNECTIONS:
                    pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
                    pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
                    cv2.line(frame, pt1, pt2, (200, 200, 200), 2)

                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

                # 2. Count extended fingers
                finger_states = count_fingers(landmarks, handedness_label)
                extended_count = sum(finger_states)
                total_fingers += extended_count

                # 3. Gesture Classification
                gesture = "Unknown"
                if extended_count == 0:
                    gesture = "Fist"
                elif extended_count == 5:
                    gesture = "Open Hand"
                elif finger_states == [0, 1, 1, 0, 0]:
                    gesture = "Peace / Victory"

                # 4. Pinch Measurement (Thumb Tip #4 to Index Tip #8)
                x1, y1 = int(landmarks[4].x * w), int(landmarks[4].y * h)
                x2, y2 = int(landmarks[8].x * w), int(landmarks[8].y * h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(frame, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (255, 0, 0), cv2.FILLED)

                pinch_dist = math.hypot(x2 - x1, y2 - y1)
                if pinch_dist < 30:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                # Draw Hand Info Tag near wrist (#0)
                wrist_x, wrist_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
                cv2.putText(frame, f"{handedness_label}: {gesture} ({extended_count})",
                            (wrist_x - 40, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Fingers Raised: {total_fingers}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hand & Gesture Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

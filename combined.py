import os
import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------------------
# 1. Environment & Path Setup
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "snapshots"), exist_ok=True)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm Base
]
TIP_IDS = [8, 12, 16, 20]

# -------------------------------------------------------------------------
# 2. Face, Eye & Smile Detector Module
# -------------------------------------------------------------------------
class AdvancedFaceDetector:
    def __init__(self):
        self.face_cascade = self._load_cascade('facedetector_frontalface_default.xml')
        self.eye_cascade = self._load_cascade('facedetector_eye.xml')
        self.smile_cascade = self._load_cascade('facedetector_smile.xml')

    def _load_cascade(self, filename):
        path = os.path.join(BASE_DIR, 'facedetector', filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing classifier: {path}")
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise FileNotFoundError(f"Failed to load classifier: {path}")
        return cascade

    def process(self, frame, gray, blur_mode=False):
        """Processes face bounding boxes, eye tracking, and expression detection."""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for i, (x, y, w, h) in enumerate(faces):
            if blur_mode:
                # Anonymize face area
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    frame[y:y+h, x:x+w] = blurred_face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(frame, "ANONYMIZED", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            else:
                # 1. Face Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 2. Eye Detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_frame = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
                )
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

                # 3. Smile & Expression Detection (Lower Face ROI)
                roi_gray_lower = roi_gray[int(h / 2):h, 0:w]
                roi_frame_lower = roi_frame[int(h / 2):h, 0:w]
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray_lower, scaleFactor=1.7, minNeighbors=22
                )

                if len(smiles) > 0:
                    expression = "Smiling :)"
                    exp_color = (0, 255, 0)
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_frame_lower, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 1)
                else:
                    expression = "Neutral / Sad :("
                    exp_color = (0, 140, 255)

                cv2.putText(frame, f"Face #{i+1} [{expression}]", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, exp_color, 2)

        return frame, faces

# -------------------------------------------------------------------------
# 3. Hand Tracker & Gesture Classifier Module
# -------------------------------------------------------------------------
class AdvancedHandTracker:
    def __init__(self, model_filename="hand_landmarker.task"):
        model_path = os.path.join(BASE_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing '{model_path}'. Download it via: "
                "wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _count_fingers(self, landmarks, handedness_label):
        fingers = []
        # Thumb: compare X coordinates based on handedness
        if handedness_label == 'Right':
            fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
        else:
            fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

        # 4 Fingers: tip Y compared to PIP joint (tip - 2)
        for tip_id in TIP_IDS:
            fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)
        return fingers

    def process(self, display_frame, raw_frame_rgb, timestamp_ms):
        h, w, _ = display_frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame_rgb)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        total_fingers = 0
        if results.hand_landmarks:
            for idx, landmarks in enumerate(results.hand_landmarks):
                handedness_label = results.handedness[idx][0].category_name

                # 1. Draw Skeleton Lines
                for p1, p2 in HAND_CONNECTIONS:
                    pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
                    pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
                    cv2.line(display_frame, pt1, pt2, (200, 200, 200), 2)

                # 2. Draw Landmark Points
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

                # 3. Finger Count & Gesture Logic
                finger_states = self._count_fingers(landmarks, handedness_label)
                extended_count = sum(finger_states)
                total_fingers += extended_count

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

                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(display_frame, (x1, y1), 6, (255, 0, 0), cv2.FILLED)
                cv2.circle(display_frame, (x2, y2), 6, (255, 0, 0), cv2.FILLED)

                pinch_dist = math.hypot(x2 - x1, y2 - y1)
                if pinch_dist < 30:
                    cv2.circle(display_frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                # 5. Hand Info Tag at Wrist (#0)
                wrist_x, wrist_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
                cv2.putText(display_frame, f"{handedness_label}: {gesture} ({extended_count})",
                            (wrist_x - 40, wrist_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display_frame, total_fingers

    def close(self):
        self.landmarker.close()

# -------------------------------------------------------------------------
# 4. Filter Utility
# -------------------------------------------------------------------------
def apply_cartoon(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray_img, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_bgr)

# -------------------------------------------------------------------------
# 5. Unified Application Loop
# -------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    face_detector = AdvancedFaceDetector()
    hand_tracker = AdvancedHandTracker("hand_landmarker.task")

    current_filter = 0
    filter_names = {
        0: "Normal",
        1: "Thermal / Heatmap",
        2: "Canny Edges",
        3: "Cartoon",
        4: "Grayscale"
    }

    enable_face = True
    enable_hands = True
    blur_mode = False
    snapshot_counter = 0
    prev_frame_time = 0

    print("\n================== Controls ==================")
    print(" [1] : Normal Vision")
    print(" [2] : Thermal / Heatmap Vision")
    print(" [3] : Canny Edge Detector")
    print(" [4] : Cartoon Filter")
    print(" [5] : Grayscale Mode")
    print(" [f] : Toggle Face Detection ON/OFF")
    print(" [h] : Toggle Hand Tracking ON/OFF")
    print(" [b] : Toggle Face Blur (Privacy)")
    print(" [s] : Save Face Snapshot")
    print(" [q] : Quit Application")
    print("==============================================\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # Baseline frames for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Apply Selected Visual Filter
        if current_filter == 1:
            display_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        elif current_filter == 2:
            canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            display_frame = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
        elif current_filter == 3:
            display_frame = apply_cartoon(frame)
        elif current_filter == 4:
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = frame.copy()

        # 2. Pipeline Execution
        faces = []
        if enable_face:
            display_frame, faces = face_detector.process(display_frame, gray, blur_mode)

        total_fingers = 0
        if enable_hands:
            timestamp_ms = int(time.time() * 1000)
            display_frame, total_fingers = hand_tracker.process(display_frame, rgb_raw, timestamp_ms)

        # 3. FPS Calculation
        now = time.time()
        fps = int(1 / (now - prev_frame_time)) if (now - prev_frame_time) > 0 else 0
        prev_frame_time = now

        # 4. HUD / Status Overlay
        cv2.putText(display_frame, f"FPS: {fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Filter: {filter_names[current_filter]}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        face_hud_col = (0, 255, 0) if enable_face else (128, 128, 128)
        cv2.putText(display_frame, f"Face [F]: {'ON' if enable_face else 'OFF'} ({len(faces)})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_hud_col, 2)

        hand_hud_col = (0, 255, 0) if enable_hands else (128, 128, 128)
        cv2.putText(display_frame, f"Hands [H]: {'ON' if enable_hands else 'OFF'} (Fingers: {total_fingers})", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_hud_col, 2)

        blur_hud_col = (0, 140, 255) if blur_mode else (180, 180, 180)
        cv2.putText(display_frame, f"Blur [B]: {'ON' if blur_mode else 'OFF'}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_hud_col, 2)

        cv2.imshow('Unified Vision Suite', display_frame)

        # 5. Keyboard Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_filter = 0
        elif key == ord('2'):
            current_filter = 1
        elif key == ord('3'):
            current_filter = 2
        elif key == ord('4'):
            current_filter = 3
        elif key == ord('5'):
            current_filter = 4
        elif key == ord('f'):
            enable_face = not enable_face
        elif key == ord('h'):
            enable_hands = not enable_hands
        elif key == ord('b'):
            blur_mode = not blur_mode
        elif key == ord('s') and len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            # Save unblurred original crop
            snapshot_img = frame[fy:fy+fh, fx:fx+fw]
            save_path = os.path.join(BASE_DIR, "snapshots", f"face_{snapshot_counter}.jpg")
            cv2.imwrite(save_path, snapshot_img)
            print(f"[+] Snapshot saved: {save_path}")
            snapshot_counter += 1

    hand_tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
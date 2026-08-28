import cv2
import time
import os

# Create snapshots directory if it doesn't exist
os.makedirs("snapshots", exist_ok=True)

# 1. Load Classifiers
face_cascade = cv2.CascadeClassifier('facedetector/facedetector_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('facedetector/facedetector_eye.xml')
smile_cascade = cv2.CascadeClassifier('facedetector/facedetector_smile.xml')

# 2. Setup Camera
cap = cv2.VideoCapture(0)

# Settings & States
blur_mode = False
prev_frame_time = 0
snapshot_counter = 0

print("\n--- Controls ---")
print("Press 'q' : Quit")
print("Press 's' : Save cropped face snapshot")
print("Press 'b' : Toggle Face Blur (Privacy Mode)")
print("----------------\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    # Pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        if blur_mode:
            # Privacy Mode: Blur the face region
            face_roi = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
            frame[y:y+h, x:x+w] = blurred_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(frame, "ANONYMIZED", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        else:
            # Normal Mode: Draw Face Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face #{i+1}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Region of Interest (ROI) inside the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect Eyes inside face
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

            # Detect Smile inside lower half of face
            roi_gray_lower = roi_gray[int(h / 2):h, 0:w]
            roi_color_lower = roi_color[int(h / 2):h, 0:w]
            smiles = smile_cascade.detectMultiScale(roi_gray_lower, scaleFactor=1.7, minNeighbors=22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color_lower, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
                cv2.putText(frame, "Smiling :)", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Overlay HUD / Stats
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    status_text = "BLUR: ON" if blur_mode else "BLUR: OFF"
    status_color = (0, 140, 255) if blur_mode else (200, 200, 200)
    cv2.putText(frame, status_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show result window
    cv2.imshow('OpenCV Vision Dashboard', frame)

    # Key Listeners
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        blur_mode = not blur_mode
    elif key == ord('s') and len(faces) > 0:
        # Save snapshot of the first detected face
        fx, fy, fw, fh = faces[0]
        face_crop = frame[fy:fy+fh, fx:fx+fw]
        snap_path = f"snapshots/face_{snapshot_counter}.jpg"
        cv2.imwrite(snap_path, face_crop)
        print(f"[+] Saved snapshot: {snap_path}")
        snapshot_counter += 1

cap.release()
cv2.destroyAllWindows()
face_cascade = cv2.CascadeClassifier('facedetector/facedetector_frontalface_default.xml')

# Verify the file loaded successfully
if face_cascade.empty():
    raise IOError("Could not load face cascade XML file. Check the file path and name.")
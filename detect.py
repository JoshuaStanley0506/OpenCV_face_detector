import os
import cv2
import time

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs("snapshots", exist_ok=True)

def load_cascade(filename):
    path = os.path.join(BASE_DIR, 'facedetector', filename)
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise FileNotFoundError(f"Missing classifier: {path}")
    return cascade

# 1. Load Classifiers
face_cascade = load_cascade('facedetector_frontalface_default.xml')
eye_cascade = load_cascade('facedetector_eye.xml')
smile_cascade = load_cascade('facedetector_smile.xml')

# 2. Camera Setup
cap = cv2.VideoCapture(0)

# Filter State Management
# Mode 0: Normal | Mode 1: Thermal | Mode 2: Edges | Mode 3: Cartoon | Mode 4: Grayscale
current_filter = 0
filter_names = {
    0: "Normal",
    1: "Thermal / Heatmap",
    2: "Canny Edges",
    3: "Cartoon",
    4: "Grayscale"
}

blur_mode = False
prev_frame_time = 0
snapshot_counter = 0

def apply_cartoon(img):
    """Creates a cel-shaded cartoon effect using bilateral filtering and edge masking."""
    # 1. Edge mask
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray_img, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    
    # 2. Color smoothing (bilateral filter keeps edges sharp)
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)
    
    # 3. Combine smoothed color with edge mask
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_bgr)
    return cartoon

print("\n================== Controls ==================")
print(" [1] : Normal Mode")
print(" [2] : Thermal / Heatmap Vision")
print(" [3] : Canny Edge Detector")
print(" [4] : Cartoon Filter")
print(" [5] : Grayscale Mode")
print(" [b] : Toggle Face Blur (Privacy)")
print(" [s] : Save Face Snapshot")
print(" [q] : Quit Application")
print("==============================================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    # Detection operates on standard grayscale for reliability
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # 1. Apply selected visual filter to base frame
    if current_filter == 1:
        # Thermal / Heatmap simulation via JET colormap
        display_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif current_filter == 2:
        # Canny edge detection converted back to 3 channels for overlay drawing
        canny_edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        display_frame = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    elif current_filter == 3:
        # Cartoon filter
        display_frame = apply_cartoon(frame)
    elif current_filter == 4:
        # Grayscale view
        display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # Normal
        display_frame = frame.copy()

    # 2. Draw detections on the filtered display frame
    for i, (x, y, w, h) in enumerate(faces):
        if blur_mode:
            # Anonymize face ROI
            face_roi = display_frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
            display_frame[y:y+h, x:x+w] = blurred_face
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(display_frame, "ANONYMIZED", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        else:
            # Face bounding box (Green)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Face #{i+1}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect Eyes (Blue) inside face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_display = display_frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_display, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

            # Detect Smiles (Red) inside lower half of face
            roi_gray_lower = roi_gray[int(h / 2):h, 0:w]
            roi_display_lower = roi_display[int(h / 2):h, 0:w]
            smiles = smile_cascade.detectMultiScale(roi_gray_lower, scaleFactor=1.7, minNeighbors=22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_display_lower, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)

    # 3. HUD / Diagnostics Overlay
    cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Faces: {len(faces)}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Filter: {filter_names[current_filter]}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    blur_text = "BLUR: ON" if blur_mode else "BLUR: OFF"
    blur_color = (0, 140, 255) if blur_mode else (180, 180, 180)
    cv2.putText(display_frame, blur_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_color, 2)

    # Render Frame
    cv2.imshow('OpenCV Face & Filter Dashboard', display_frame)

    # 4. Keyboard Controls
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
    elif key == ord('b'):
        blur_mode = not blur_mode
    elif key == ord('s') and len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        snapshot_img = frame[fy:fy+fh, fx:fx+fw]
        path = f"snapshots/face_{snapshot_counter}.jpg"
        cv2.imwrite(path, snapshot_img)
        print(f"[+] Snapshot saved: {path}")
        snapshot_counter += 1

cap.release()
cv2.destroyAllWindows()
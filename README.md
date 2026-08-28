# 👁️ Real-Time Multi-Vision Suite (OpenCV & MediaPipe)

A modular, real-time computer vision application combining classical feature cascades (**OpenCV**) with modern spatial landmark detection (**MediaPipe Tasks API**). The suite provides face and eye tracking, expression analysis, real-time hand skeleton tracking, gesture recognition, privacy face anonymization, and live visual shaders.

---

## ✨ Features

- **Face & Eye Detection**: Haar feature cascade detection with cropped Region-of-Interest (ROI) eye tracking.
- **Expression Classifier**: Real-time mouth curvature heuristic to classify expressions (`Smiling :)` vs `Neutral/Sad :(`).
- **Privacy Mode**: Dynamic Gaussian blur anonymization for all detected faces.
- **Hand Landmark Tracking**: 21-point spatial mesh tracking powered by the MediaPipe Tasks API.
- **Gesture & Pinch Detection**:
  - Live finger extension counter (0 to 5 per hand).
  - Gesture recognition (`Fist`, `Open Hand`, `Peace / Victory`).
  - Euclidean distance-based dynamic pinch gauge (Thumb to Index).
- **Visual Shaders & Filters**: Live switching between Normal, Thermal/Heatmap (JET Colormap), Canny Edge Detection, Bilateral Cartoon Filter, and Grayscale modes.
- **Instant Snapshot**: Save unblurred face crops to `snapshots/` at the press of a key.

---
## 🤝 Authorship & Acknowledgements

- **Joshua Stanley George** — Project design, integration, testing, code review, and conceptual understanding.
- **AI Collaboration (Google Antigravity)** — Assisted with the implementation details, MediaPipe 1.0 Tasks API migration, and pipeline consolidation.

> *"Built through iterative human-AI collaboration — with code thoroughly studied, debugged, and reviewed for production understanding."*

## 🛠️ Project Structure

```text
opencv-haarcascades/
├── combined.py             # Unified real-time pipeline (Face + Hands + Shaders)
├── detect.py               # Standalone Haar cascade face detector & filter suite
├── hand_tracker.py         # Standalone MediaPipe hand landmarker & gesture tracker
├── facedetector/           # Haar cascade XML model definitions
│   ├── facedetector_frontalface_default.xml
│   ├── facedetector_eye.xml
│   └── facedetector_smile.xml
├── snapshots/              # Captured face images
├── hand_landmarker.task    # MediaPipe hand tracking binary model
└── README.md               # Documentation

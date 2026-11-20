# Naruto Jutsu Tracker

Real-time hand gesture recognition that triggers Naruto-inspired visual and sound effects using your webcam.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0-orange)

## Jutsu Gestures

| Gesture | Jutsu | Effect |
|---------|-------|--------|
| One open palm | **Rasengan** | Spinning blue energy sphere with particle swirls |
| Both palms open | **Rasenshuriken** | Rasengan core + 4 spinning wind blades |
| Closed fist | **Chidori** | Crackling lightning bolts with electric sparks |
| Wink (one eye closed) | **Amaterasu** | Black flames engulfing the screen + Mangekyo Sharingan |

All effects include **synthesized sound effects** — no external audio files needed.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MediaPipe models

Download both model files into the project directory:

```bash
# Hand landmark model
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Face landmark model (for eye/wink detection)
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 3. Run

```bash
python naruto_jutsu.py
```

## Controls

- **R** — Reset all active effects
- **Q** — Quit

## How It Works

- **Hand Detection**: MediaPipe Hand Landmarker (VIDEO mode) detects hand landmarks in real-time
- **Face Detection**: MediaPipe Face Landmarker tracks eye landmarks for wink/Amaterasu detection
- **Gesture Classification**: Finger states (extended vs curled) determine open palm vs fist; Eye Aspect Ratio (EAR) detects winks
- **Visual Effects**: OpenCV-drawn particle systems, energy spheres, lightning bolts, and flame overlays
- **Sound Effects**: Real-time synthesized audio via `sounddevice` — spinning hums, wind noise, electric crackling, and deep rumbles

## Requirements

- Python 3.10+
- Webcam
- Audio output device (speakers/headphones)

## License

MIT

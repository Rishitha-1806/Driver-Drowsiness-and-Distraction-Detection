# Driver Drowsiness and Distraction Detection System

## Overview

This project is an AI-powered Driver Monitoring System designed to improve road safety by detecting driver drowsiness and distraction in real time. The system combines Computer Vision, Deep Learning, and Machine Learning techniques to monitor driver behavior and generate alerts whenever unsafe conditions are detected.

The application uses facial landmark detection for eye and mouth analysis, YOLOv8 for mobile phone detection, and real-time audio alerts to warn drivers.

---

## Features

### Drowsiness Detection

* Detects eye closure using Eye Aspect Ratio (EAR).
* Uses Dlib's 68 facial landmark model.
* Triggers an alert when eyes remain closed for a specified duration.

### Yawn Detection

* Detects yawning by measuring the distance between upper and lower lips.
* Tracks yawning frequency.
* Generates alerts when excessive yawning is detected.

### Mobile Phone Usage Detection

* Uses YOLOv8 object detection model.
* Detects mobile phone usage while driving.
* Triggers alerts for continuous phone usage.

### No Face Detection

* Uses Haar Cascade face detection.
* Alerts when the driver's face is not visible.

### Real-Time Alerts

* Audio warning system.
* Visual alerts displayed on screen.
* Multi-threaded alarm execution.

---

## Technologies Used

* Python
* OpenCV
* Dlib
* YOLOv8
* NumPy
* Imutils
* Haar Cascade Classifier
* Multithreading
* Computer Vision
* Deep Learning

---

## Project Structure

```text
DriverDrowsinessAndDistraction/
│
├── app.py
├── alarm.wav
├── haarcascade_frontalface_default.xml
├── shape_predictor_68_face_landmarks.dat
├── yolov8n.pt
├── requirements.txt
└── README.md
```

---

## Required Files

The following pretrained model files are not included in this repository because of their large size.

### 1. Dlib Facial Landmark Predictor

Download:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 or https://www.kaggle.com/datasets/playmakerg/shape-predictor-68-face-landmarksdat?resource=download

After extraction, place:

```text
shape_predictor_68_face_landmarks.dat
```

inside the project directory.

---

### 2. YOLOv8 Model

Download:

https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Place:

```text
yolov8n.pt
```

inside the project directory.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Rishitha-1806/Driver-Drowsiness-and-Distraction-Detection.git

cd Driver-Drowsiness-and-Distraction-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Additional Packages

```bash
pip install ultralytics
pip install dlib
pip install opencv-python
pip install imutils
pip install playsound
```

---

## Running the Project

```bash
python app.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
```

Example:

```bash
python app.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
```

---

## Detection Workflow

1. Webcam captures live video stream.
2. Haar Cascade detects face presence.
3. Dlib identifies 68 facial landmarks.
4. Eye Aspect Ratio (EAR) is calculated.
5. Drowsiness is detected when EAR falls below threshold.
6. Mouth landmarks are analyzed for yawning detection.
7. YOLOv8 detects mobile phone usage.
8. Audio and visual alerts are triggered when unsafe behavior is identified.

---

## Alert Conditions

| Event            | Detection Method            |
| ---------------- | --------------------------- |
| Drowsiness       | Eye Aspect Ratio (EAR)      |
| Yawning          | Lip Distance Analysis       |
| Mobile Usage     | YOLOv8 Object Detection     |
| No Face Detected | Haar Cascade Face Detection |

---

## Future Enhancements

* Head Pose Estimation
* Seatbelt Detection
* Driver Identity Verification
* Cloud Dashboard Monitoring
* Edge Device Deployment
* Fatigue Prediction Analytics

---

## Applications

* Smart Vehicles
* Driver Safety Systems
* Fleet Monitoring
* Transportation Industry
* Advanced Driver Assistance Systems (ADAS)

---
## How to Run the Project

### Step 1: Activate Virtual Environment

```bash
activate major1
```

### Step 2: Navigate to Project Directory

```bash
cd C:\Users\MERUGUMALA RISHITHA\OneDrive\Desktop\major\DriverDrowsinessAndDistraction
```

### Step 3: Run the Application

```bash
python app3.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
```

### Command Explanation

* `app3.py` → Main application file.
* `--shape-predictor` → Path to the Dlib facial landmark model.
* `shape_predictor_68_face_landmarks.dat` → Facial landmark predictor file used for eye and mouth detection.
* `--alarm alarm.wav` → Audio alert played when drowsiness, yawning, mobile usage, or no-face conditions are detected.

### Exit the Application

Press:

```text
q
```

to stop the video stream and close the application.

## Author

**Merugumala Rishitha**

B.Tech - Computer Science and Engineering

TKR College of Engineering and Technology

Hyderabad, India

---





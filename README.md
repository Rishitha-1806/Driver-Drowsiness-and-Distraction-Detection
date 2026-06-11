# Driver Drowsiness and Distraction Detection

An AI-powered driver monitoring system that detects drowsiness and distracted driving behaviors in real time using Computer Vision, Deep Learning, and Machine Learning techniques. The system provides instant alerts to improve road safety and reduce accident risks.

## Features

* Real-time driver drowsiness detection using Eye Aspect Ratio (EAR)
* Facial landmark detection with Dlib
* Driver distraction detection using YOLOv8
* Audio and visual alerts for unsafe driving behavior
* Real-time webcam monitoring
* Deep learning-based classification for enhanced accuracy

## Tech Stack

* Python
* OpenCV
* TensorFlow
* Keras
* YOLOv8
* Dlib
* NumPy
* Docker

## Required Model Files

The following model files are not included in this repository due to their large size.

### 1. Dlib Facial Landmark Model

Download:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 or https://www.kaggle.com/datasets/playmakerg/shape-predictor-68-face-landmarksdat?resource=download

Extract the archive and place:

shape_predictor_68_face_landmarks.dat

in the project root directory.

### 2. YOLOv8 Model

Download:

https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Place:

yolov8n.pt

in the project root directory.

## Installation

1. Clone the repository

```bash
git clone https://github.com/Rishitha-1806/Driver-Drowsiness-and-Distraction-Detection.git
cd Driver-Drowsiness-and-Distraction-Detection
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the required model files and place them in the project directory.

4. Run the application

```bash
app3.py
```

## Project Workflow

1. Webcam captures driver's face in real time.
2. Facial landmarks are detected using Dlib.
3. Eye Aspect Ratio (EAR) is calculated to identify drowsiness.
4. YOLOv8 detects distraction activities such as mobile phone usage.
5. The system generates instant alerts when unsafe conditions are detected.

## Future Enhancements

* Head pose estimation
* Seatbelt detection
* Driver identification
* Cloud-based monitoring dashboard
* Edge deployment for embedded systems

## Author

Merugumala Rishitha

B.Tech Computer Science and Engineering

TKR College of Engineering and Technology


# USAGE
# python app3.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python app3.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
# from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os

# ------------------------------------------------------------
# YOLOv8 for mobile phone detection
# ------------------------------------------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics YOLO not installed. Mobile detection disabled.")

# ------------------------------------------------------------
def sound_alarm(path):
    """Play an alarm sound in a separate thread."""
    playsound.playsound(path)


def euclidean(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)
# ------------------------------------------------------------
# construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
# Mobile detection arguments (now used for YOLO)
ap.add_argument("-mc", "--mobile-confidence", type=float, default=0.5,
                help="confidence threshold for YOLO mobile detection")
ap.add_argument("-mf", "--mobile-frames", type=int, default=10,
                help="number of consecutive mobile detections to trigger alarm")
args = vars(ap.parse_args())

# ------------------------------------------------------------
# Landmark helper functions (unchanged)
# ------------------------------------------------------------
def get_landmarks(im):
    # dlib expects gray or RGB uint8
    if im is None or im.size == 0:
        return "error"

    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    rgb_im = np.ascontiguousarray(rgb_im, dtype=np.uint8)

    rects = detector(rgb_im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(rgb_im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    """Draw landmark points on the image."""
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    """Get y-coordinate of top lip center."""
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])

def bottom_lip(landmarks):
    """Get y-coordinate of bottom lip center."""
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])

def mouth_open(image):
    """Detect yawn by measuring lip distance."""
    landmarks = get_landmarks(image)
    if landmarks == "error":
        return image, 0
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

# ------------------------------------------------------------
# Constants for alarms
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 20
MOBILE_CONSEC_FRAMES = args["mobile_frames"]
MOBILE_CONF_THRESH = args["mobile_confidence"]

# Counters and flags for each alarm type
COUNTER = 0                     # drowsiness consecutive frames
MOBILE_COUNTER = 0               # mobile consecutive detections

drowsy_alarm_on = False
yawn_alarm_on = False
no_face_alarm_on = False
mobile_alarm_on = False

yawns = 0
yawn_status = False

# ------------------------------------------------------------
# Load dlib face detector and landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Get indexes for left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ------------------------------------------------------------
# Load YOLOv8 model for mobile phone detection
mobile_model = None
if YOLO_AVAILABLE:
    # Use the smallest YOLOv8 model (nano) – it will be downloaded automatically if missing
    model_path = "yolov8n.pt"
    print("[INFO] loading YOLOv8 model for mobile detection...")
    try:
        mobile_model = YOLO(model_path)   # automatically downloads if not present
        print("[INFO] YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {e}")
        mobile_model = None
else:
    print("[INFO] YOLOv8 not available; mobile detection disabled.")

# ------------------------------------------------------------
# Start video stream
print("[INFO] starting video stream thread...")
vs = VideoStream(0).start()
time.sleep(1.0)

# Load Haar cascade for face detection (used for no‑face alarm)
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)

# ------------------------------------------------------------
# Main loop
while True:
    # Read a single frame and use it for all detections
    frame = vs.read()
    if frame is None:
        continue

    frame = imutils.resize(frame, width=450)

    # Force correct dtype + contiguous memory (dlib requirement)
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    # Detect faces using Haar cascade (for no‑face alarm and optional region)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    print("Detected Faces:", len(faces))

    # ------------------- NO FACE ALARM -------------------
    if len(faces) == 0:
        if not no_face_alarm_on:
            no_face_alarm_on = True
            if args["alarm"] != "":
                t = Thread(target=sound_alarm, args=(args["alarm"],))
                t.daemon = True
                t.start()
        # No faces, skip the rest of the processing (eyes/yawn/mobile)
        cv2.putText(frame, "NO FACE DETECTED", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Face present – reset no‑face alarm flag
        no_face_alarm_on = False
        
        # ------------------- DROWSINESS DETECTION (eyes) -------------------
        # dlib expects either 8-bit grayscale or RGB (uint8). Use RGB for robustness.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        print("rgb dtype:", rgb.dtype, "shape:", rgb.shape, "contig:", rgb.flags['C_CONTIGUOUS'])
        
        rects = detector(rgb, 0)
        for rect in rects:
            shape = predictor(rgb, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Drowsiness logic
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not drowsy_alarm_on:
                    drowsy_alarm_on = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                drowsy_alarm_on = False

            cv2.putText(frame, "EAR: {:.2f} Count {}".format(ear, COUNTER),
                        (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ------------------- MOBILE USAGE DETECTION (YOLOv8) -------------------
        if mobile_model is not None:
            # Run YOLOv8 inference on the whole frame
            results = mobile_model(frame, verbose=False)[0]   # get first result
            # COCO class 67 = cell phone
            mobile_detected = False
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 67 and conf >= MOBILE_CONF_THRESH:
                    mobile_detected = True
                    # Draw bounding box for visualization
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            if mobile_detected:
                MOBILE_COUNTER += 1
                if MOBILE_COUNTER >= MOBILE_CONSEC_FRAMES and not mobile_alarm_on:
                    mobile_alarm_on = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                    cv2.putText(frame, "MOBILE USAGE ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                MOBILE_COUNTER = max(0, MOBILE_COUNTER - 1)
                if MOBILE_COUNTER == 0:
                    mobile_alarm_on = False

        # ------------------- YAWN DETECTION -------------------
        # Use the same frame for yawn analysis
        image_landmarks, lip_distance = mouth_open(frame)

        prev_yawn_status = yawn_status
        if lip_distance > 25:
            yawn_status = True
            cv2.putText(frame, "Subject is Yawning", (50, 450),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            output_text = " Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        else:
            yawn_status = False

        # Count a yawn when transition from True to False
        if prev_yawn_status and not yawn_status:
            yawns += 1

        # Trigger alarm after 10 yawns (reset counter and alarm flag afterwards)
        if yawns >= 10:
            if not yawn_alarm_on:   # play only once per 10 yawns
                yawn_alarm_on = True
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
            # Reset counter and allow future alarms
            yawns = 0
            yawn_alarm_on = False

        # Show the landmark image separately
        cv2.imshow('Live Landmarks', image_landmarks)

    # Show the main frame with all annotations
    cv2.imshow("Frame", frame)

    # Key press check (placed outside the if-else so 'q' works even with no face)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
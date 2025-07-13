import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from ultralytics import YOLO  # YOLOv8
import time

# Load gesture model
model = load_model('gesture_model.h5')

# YOLO model (v8n is fast, you can use v8s or v8m for better accuracy)
yolo_model = YOLO('yolov8n.pt')

# Gesture labels
DATA_PATH = 'gesture_data'
actions = np.array(os.listdir(DATA_PATH))
threshold = 0.8

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Buffers
sequence = []
sentence_buffer = []
last_gesture = ''
last_object = ''
object_cooldown = 1.5  # seconds
last_object_time = 0

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, lh, rh]).flatten()

# Webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Gesture Detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            confidence = res[np.argmax(res)]

            if confidence > threshold:
                gesture = actions[np.argmax(res)]
                if gesture != last_gesture:
                    last_gesture = gesture
                    sentence_buffer.append(gesture)
                    print("Detected Gesture:", gesture)

        # Object Detection
        now = time.time()
        results_yolo = yolo_model.predict(source=frame, conf=0.5, iou=0.4, verbose=False)[0]
        for box in results_yolo.boxes:
            cls = int(box.cls[0])
            label = yolo_model.model.names[cls]

            if label != "person" and label != last_object and now - last_object_time > object_cooldown:
                last_object = label
                last_object_time = now
                sentence_buffer.append(label)
                print("Detected Object:", label)
                break  # Only one object per frame

        # Display Sentence
        display_text = ' '.join(sentence_buffer)
        cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(image, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show result
        cv2.imshow('Gesture + Object Detection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sentence_buffer.clear()
            last_gesture = ''
            last_object = ''
            print("🔁 Sentence reset")

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load model and actions
model = load_model('gesture_model.h5')
DATA_PATH = 'gesture_data'
actions = np.array(os.listdir(DATA_PATH))
threshold = 0.8

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence = []

# Extract keypoints from frame
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# Setup GUI
root = tk.Tk()
root.title("Gesture Recognition")
root.geometry("800x600")

label = tk.Label(root, text="Gesture: ", font=('Helvetica', 24))
label.pack()

canvas = tk.Label(root)
canvas.pack()

cap = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def update():
    global sequence

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        if res[np.argmax(res)] > threshold:
            predicted_action = actions[np.argmax(res)]
            label.config(text=f"Gesture: {predicted_action}")

    # Convert to ImageTk for GUI
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)
    root.after(10, update)

root.after(0, update)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()

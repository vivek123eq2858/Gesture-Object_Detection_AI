import cv2
import numpy as np
import os
import time
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Paths
DATA_PATH = os.path.join('gesture_data')
os.makedirs(DATA_PATH, exist_ok=True)

# User input
action = input("Enter the title of the gesture/action you want to train: ").strip()
num_sequences = 30
sequence_length = 30

# Create folders
for seq in range(num_sequences):
    os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Global control flags
cap = cv2.VideoCapture(0)

# Tkinter UI
root = tk.Tk()
root.title("Gesture Recorder")
root.geometry("900x700")

label = tk.Label(root, text=f"Preparing to record: '{action}'", font=('Helvetica', 18))
label.pack(pady=10)

canvas = tk.Label(root)
canvas.pack()

# Record gestures after delay
def record_gestures():
    time.sleep(5)  # Delay after window open
    label.config(text=f"Recording started for: '{action}'")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sequence in range(num_sequences):
            print(f"Recording Sequence {sequence + 1}/{num_sequences}")
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Only update UI every 2 frames for speed boost
                if frame_num % 2 == 0:
                    cv2.putText(image, f'{action} | Seq {sequence + 1}/{num_sequences}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    imgtk = ImageTk.PhotoImage(image=img)
                    canvas.imgtk = imgtk
                    canvas.configure(image=imgtk)
                    canvas.update()

    label.config(text="✅ Recording Complete")
    cap.release()
    cv2.destroyAllWindows()

# Launch recording in thread
def start():
    threading.Thread(target=record_gestures).start()

root.after(100, start)
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()

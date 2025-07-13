import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Directory where data is saved
DATA_PATH = 'gesture_data'

# Get all action labels (folder names)
actions = np.array(os.listdir(DATA_PATH))
print("Detected Actions:", actions)

sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

# Load data
for action in actions:
    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame_num in range(30):  # assuming 30 frames per sequence
            npy_path = os.path.join(DATA_PATH, action, sequence, f"{frame_num}.npy")
            res = np.load(npy_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to arrays
X = np.array(sequences)  # Shape: (num_samples, 30, num_features)
y = to_categorical(labels).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Data Loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)

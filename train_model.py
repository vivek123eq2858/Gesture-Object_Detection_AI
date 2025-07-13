import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --------- Load and Preprocess Data ---------
DATA_PATH = 'gesture_data'
actions = np.array(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame_num in range(30):
            path = os.path.join(DATA_PATH, action, sequence, f"{frame_num}.npy")
            res = np.load(path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --------- Build and Train the Model ---------
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_model.h5')
print("✅ Model saved as 'gesture_model.h5'")

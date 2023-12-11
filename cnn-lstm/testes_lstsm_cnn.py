import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed

# Ekstraksi Frame dari Video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    count = 0

    while success:
        frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, image)
        success, image = cap.read()
        count += 1

# Resize Frame
def resize_frame(frame_path, target_width, target_height):
    img = cv2.imread(frame_path)
    img_resized = cv2.resize(img, (target_width, target_height))
    cv2.imwrite(frame_path, img_resized)

# Buat Dataset
def create_dataset(input_folder, sequence_length, target_width, target_height):
    X = []
    y = []

    for video_folder in os.listdir(input_folder):
        video_folder_path = os.path.join(input_folder, video_folder)
        if os.path.isdir(video_folder_path):
            frames = []

            for frame_name in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, frame_name)
                resize_frame(frame_path, target_width, target_height)
                img = cv2.imread(frame_path)
                frames.append(img)

            if len(frames) >= sequence_length:
                X.append(frames[:sequence_length])
                y.append(int(video_folder.split("_")[0]))  # Ambil label dari nama folder

    return np.array(X), np.array(y)

# Arsitektur CNN
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu')
])

# Arsitektur LSTM
lstm_model = Sequential([
    TimeDistributed(cnn_model, input_shape=(10, 64, 64, 3)),
    LSTM(50, return_sequences=True),
    Flatten(),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.utils import plot_model
plot_model(lstm_model, to_file='model_plot2.png', show_shapes=True, show_layer_names=True)

# Ekstraksi Frame, Resize, dan Buat Dataset
PARENT_DIRECTORY = os.getcwd()
DATASET_PATH = os.path.join(PARENT_DIRECTORY, 'cnn-lstm', 'new-dataset')

video_path = "/path/to/your/video.mp4"
output_folder = os.path.join(DATASET_PATH,'1')
sequence_length = 10
target_width = 64
target_height = 64

# extract_frames(video_path, output_folder)
X, y = create_dataset(output_folder, sequence_length, target_width, target_height)

# Pisahkan Data untuk Pelatihan dan Pengujian
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih Model
lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

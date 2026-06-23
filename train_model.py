import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

DATASET_PATH = "leapGestRecog"

images = []
labels = []
gesture_names = []

print("Loading Dataset...")

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if gesture not in gesture_names:
            gesture_names.append(gesture)

        label = gesture_names.index(gesture)

        for image_name in os.listdir(gesture_path):

            image_path = os.path.join(
                gesture_path,
                image_name
            )

            img = cv2.imread(image_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            images.append(img)
            labels.append(label)

X = np.array(images) / 255.0
y = np.array(labels)

y = to_categorical(y)

print("Dataset Loaded")
print("Images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(64,64,3)
    )
)

model.add(MaxPooling2D(2,2))

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.3))

model.add(
    Dense(
        y.shape[1],
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

model.save("gesture_model.keras")

np.save(
    "gesture_names.npy",
    gesture_names
)

print("Model Saved Successfully")
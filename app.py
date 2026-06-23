import streamlit as st
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = load_model("gesture_model.keras")

gesture_names = np.load(
    "gesture_names.npy",
    allow_pickle=True
)

# Model ke output classes jitni hain utni hi names use karo
num_classes = model.output_shape[1]

if len(gesture_names) > num_classes:
    gesture_names = gesture_names[:num_classes]

# ---------------- UI ----------------

st.title("✋ Hand Gesture Recognition")

st.write(
    "Upload a hand gesture image and predict the gesture using CNN."
)

st.sidebar.header("Model Information")

st.sidebar.write("Algorithm : CNN")
st.sidebar.write("Dataset : LeapGestRecog")
st.sidebar.write(f"Classes : {num_classes}")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    img = np.array(image)

    # Grayscale image
    if len(img.shape) == 2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    # PNG with alpha channel
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2RGB
        )

    img = cv2.resize(
        img,
        (64, 64)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    st.write("Input Shape:", img.shape)

    prediction = model.predict(
        img,
        verbose=0
    )

    confidence = float(
        np.max(prediction)
    ) * 100

    gesture = gesture_names[
        np.argmax(prediction)
    ]

    st.subheader("Prediction Result")

    if confidence < 70:

        st.error(
            "❌ Unable to recognize gesture confidently"
        )

    else:

        st.success(
            f"✅ Predicted Gesture : {gesture}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )

    st.subheader(
        "Prediction Probabilities"
    )

    for i in range(num_classes):

        prob = float(
            prediction[0][i]
        ) * 100

        st.write(
            f"{gesture_names[i]} : {prob:.2f}%"
        )

# ---------------- FOOTER ----------------

st.markdown("---")

st.write(
    "Hand Gesture Recognition using CNN and LeapGestRecog Dataset"
)
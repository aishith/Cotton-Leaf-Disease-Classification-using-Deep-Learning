import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["diseased cotton leaf", "diseased cotton plant", "fresh cotton leaf"]
CNN_INPUT_SIZE = (128, 128)
VGG16_INPUT_SIZE = (224, 224)


@st.cache_resource
def load_models():
    """Load both models once and cache them across reruns/users."""
    cnn_path = os.path.join(BASE_DIR, "model_cnn.h5")
    vgg16_path = os.path.join(BASE_DIR, "model_vgg16.h5")
    cnn = load_model(cnn_path) if os.path.exists(cnn_path) else None
    vgg16 = load_model(vgg16_path) if os.path.exists(vgg16_path) else None
    return cnn, vgg16


def preprocess_image(image, target_size):
    img = image.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


st.set_page_config(page_title="Cotton Leaf Disease Detection", page_icon="🌱")

st.title("🌱 Cotton Leaf Disease Detection")
st.write("Upload a photo of a cotton leaf or plant to check its condition.")

cnn_model, vgg16_model = load_models()

algo = st.selectbox("Model", ["CNN", "VGG16"])
uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        model = vgg16_model if algo == "VGG16" else cnn_model
        target_size = VGG16_INPUT_SIZE if algo == "VGG16" else CNN_INPUT_SIZE

        if model is None:
            st.error(f"{algo} model file not found. Make sure model_{algo.lower()}.h5 is in the app folder.")
        else:
            batch = preprocess_image(image, target_size)
            preds = model.predict(batch)
            class_index = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds)) * 100

            st.success(f"**{CLASS_NAMES[class_index]}**")
            st.write(f"Confidence: {confidence:.1f}% · Model: {algo}")

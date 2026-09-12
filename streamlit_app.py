import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ["diseased cotton leaf", "diseased cotton plant", "fresh cotton leaf"]
CNN_INPUT_SIZE = (128, 128)
VGG16_INPUT_SIZE = (224, 224)

# Friendlier labels + short advice shown alongside the raw class name
DISPLAY_INFO = {
    "diseased cotton leaf": {
        "label": "Diseased Leaf",
        "status": "warning",
        "note": "Signs of disease detected on the leaf. Consider isolating the plant and consulting an agronomist.",
    },
    "diseased cotton plant": {
        "label": "Diseased Plant",
        "status": "error",
        "note": "The plant shows signs of disease. Early treatment can help limit spread to nearby crops.",
    },
    "fresh cotton leaf": {
        "label": "Healthy Leaf",
        "status": "success",
        "note": "No visible signs of disease. Keep monitoring regularly.",
    },
}


@st.cache_resource
def load_models():
    cnn_path = os.path.join(BASE_DIR, "model_cnn.h5")
    vgg16_path = os.path.join(BASE_DIR, "model_vgg16.h5")
    cnn = load_model(cnn_path) if os.path.exists(cnn_path) else None
    vgg16 = load_model(vgg16_path) if os.path.exists(vgg16_path) else None
    return cnn, vgg16


def preprocess_image(image, target_size):
    img = image.convert("RGB").resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light custom styling on top of the theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle { color: #666; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .result-card {
        border: 1px solid #e3e6e0;
        border-radius: 12px;
        padding: 1.5rem;
        background: #f9faf8;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

cnn_model, vgg16_model = load_models()

# ---------------------------------------------------------------------------
# Sidebar — model choice + project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    algo = st.radio("Prediction model", ["CNN", "VGG16"], help="VGG16 is a larger, transfer-learning model; CNN is a lighter model trained from scratch.")

    st.divider()
    st.subheader("About")
    st.write(
        "This tool classifies cotton leaf/plant images into three categories: "
        "healthy, diseased leaf, or diseased plant, using deep learning models "
        "trained on a labeled cotton crop dataset."
    )
    st.caption("Capstone project · CNN & VGG16 transfer learning")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🌱 Cotton Leaf Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a photo of a cotton leaf or plant to check its condition.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")
    else:
        st.info("Upload a JPG or PNG image to get started.")
        predict_clicked = False

with col2:
    st.subheader("Result")
    if uploaded_file is None:
        st.markdown(
            '<div class="result-card">Prediction will appear here once you upload an image and click Predict.</div>',
            unsafe_allow_html=True,
        )
    elif predict_clicked:
        model = vgg16_model if algo == "VGG16" else cnn_model
        target_size = VGG16_INPUT_SIZE if algo == "VGG16" else CNN_INPUT_SIZE

        if model is None:
            st.error(f"{algo} model file not found. Make sure model_{algo.lower()}.h5 is in the app folder.")
        else:
            with st.spinner(f"Running {algo} model..."):
                batch = preprocess_image(image, target_size)
                preds = model.predict(batch)
                class_index = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds)) * 100
                predicted_class = CLASS_NAMES[class_index]
                info = DISPLAY_INFO[predicted_class]

            getattr(st, info["status"])(f"**{info['label']}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(min(int(confidence), 100))
            st.caption(f"Model used: {algo}")
            st.write(info["note"])
    else:
        st.markdown(
            '<div class="result-card">Image ready — click <b>Predict</b> to classify it.</div>',
            unsafe_allow_html=True,
        )

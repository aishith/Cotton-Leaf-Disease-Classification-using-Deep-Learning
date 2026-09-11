import os

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# Class order MUST match the order Keras assigned during training.
# ImageDataGenerator.flow_from_directory() sorts class folders alphabetically,
# so for this dataset that is:
CLASS_NAMES = ["diseased cotton leaf", "diseased cotton plant", "fresh cotton leaf"]

# Models are loaded once at startup, not per-request.
CNN_PATH = os.path.join(BASE_DIR, "model_cnn.h5")
VGG16_PATH = os.path.join(BASE_DIR, "model_vgg16.h5")

cnn_model = load_model(CNN_PATH) if os.path.exists(CNN_PATH) else None
vgg16_model = load_model(VGG16_PATH) if os.path.exists(VGG16_PATH) else None

CNN_INPUT_SIZE = (128, 128)
VGG16_INPUT_SIZE = (224, 224)


def preprocess_image(file_storage, target_size):
    """Load an uploaded image and prepare it the same way it was prepared during training."""
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0  # matches ImageDataGenerator(rescale=1./255)
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please choose an image file first.")

    file = request.files["file"]
    algo = request.form.get("algo", "cnn")

    try:
        if algo == "vgg16":
            if vgg16_model is None:
                return render_template("index.html", error="VGG16 model file not found on server.")
            batch = preprocess_image(file, VGG16_INPUT_SIZE)
            preds = vgg16_model.predict(batch)
        else:
            if cnn_model is None:
                return render_template("index.html", error="CNN model file not found on server.")
            batch = preprocess_image(file, CNN_INPUT_SIZE)
            preds = cnn_model.predict(batch)

        class_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        result = CLASS_NAMES[class_index]

    except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a stack trace
        return render_template("index.html", error=f"Couldn't process that image: {exc}")

    return render_template("result.html", result=result, confidence=round(confidence * 100, 1), algo=algo.upper())


if __name__ == "__main__":
    # Local dev only. In production this is run via gunicorn (see Procfile).
    app.run(debug=True, host="0.0.0.0", port=8000)

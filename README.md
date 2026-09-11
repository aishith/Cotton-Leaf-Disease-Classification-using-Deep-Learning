# Cotton Leaf Disease Detection

A web app that classifies cotton leaf/plant images as healthy or diseased,
using two deep learning models: a CNN trained from scratch and a VGG16
transfer-learning model. Built with Flask.

## Project structure

```
cotton-app/
├── train_model.ipynb   # Trains both models on the dataset
├── app.py              # Flask app — loads the trained models, serves predictions
├── requirements.txt
├── Procfile
├── .gitignore
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    └── style.css
```

## Dataset

3 classes: `diseased cotton leaf`, `diseased cotton plant`, `fresh cotton leaf`.
Organize images as:

```
datasets/Cotton Disease/
├── train/<class>/*.jpg
├── val/<class>/*.jpg
└── test/<class>/*.jpg
```

## Training

1. Open `train_model.ipynb` in Jupyter, VSCode, or Google Colab.
2. Point `DATA_DIR` at your dataset folder.
3. Run all cells — this produces `model_cnn.h5` and `model_vgg16.h5`.
4. Copy both files into the project root, next to `app.py`.

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`, upload a leaf image, pick a model, and view the
prediction.

## Deployment

Deployed on [Render](https://render.com):

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --timeout 120`

## Tech stack

Python, TensorFlow/Keras, Flask, HTML/CSS.

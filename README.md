# Cotton Leaf Disease Detection — Flask App

A clean rebuild of the cotton crop disease detection project: a CNN and a VGG16
transfer-learning model, served through a small Flask web app.

## Project structure

```
cotton-app/
├── train_model.ipynb   # Retrains both models from your dataset
├── app.py              # Flask app (loads the trained models, serves predictions)
├── requirements.txt
├── Procfile             # For deployment (gunicorn)
├── .gitignore
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    └── style.css
```

## 1. Train the models

1. Open `train_model.ipynb` in Jupyter, VSCode, or upload it to **Google Colab**
   (recommended if you don't have a GPU — free GPU under
   Runtime > Change runtime type > GPU).
2. Place your `datasets/Cotton Disease/` folder (with `train/`, `val/`, `test/`
   subfolders, one folder per class) next to the notebook, or update `DATA_DIR`
   at the top of the notebook to point to it.
3. Run all cells. This produces `model_cnn.h5` and `model_vgg16.h5`.
4. Copy both `.h5` files into this `cotton-app/` folder, next to `app.py`.

**Class order matters.** `flow_from_directory` assigns class indices
alphabetically. For this dataset that's:
`0 = diseased cotton leaf, 1 = diseased cotton plant, 2 = fresh cotton leaf`.
`app.py` already assumes this order — if you rename folders or add classes,
update `CLASS_NAMES` in `app.py` to match.

## 2. Run the app locally

```bash
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`, upload a leaf image, pick a model, and check the
prediction.

## 3. Deploy it

1. Push this folder to a GitHub repo (the `.gitignore` already excludes
   `datasets/`, `__pycache__/`, and virtual environments — keep the two `.h5`
   model files, they're needed at runtime).
2. Create a free account at [render.com](https://render.com), click
   **New Web Service**, and connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --timeout 120`
   (the longer timeout gives TensorFlow room to load the VGG16 model on
   startup and handle slower first predictions).
5. Deploy. Render gives you a public URL once the build finishes.

**Note on model size:** `model_vgg16.h5` will likely be 50–90MB depending on
how you save it. That's under GitHub's 100MB hard limit, so a normal `git push`
should work. If it grows past that, use Git LFS (`git lfs track "*.h5"`).

## Notes on what changed from the original project

- Fixed the hardcoded Windows path (`C:\Users\...`) that broke the app outside
  the original developer's machine — model paths are now relative to the app
  folder.
- Corrected `CLASS_NAMES` to 3 classes, matching the actual dataset (the
  original code had a stray 4th class that didn't exist in the data).
- Dropped Django in favor of Flask, which needs far less production
  configuration (no `SECRET_KEY`/`ALLOWED_HOSTS`/`collectstatic` setup) for a
  project this size.


from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "breast_lr.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    load_err = f"Could not load model from {MODEL_PATH}: {e}"

DEFAULT_FEATURES = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

def get_feature_order():
    names = getattr(model, "feature_names_in_", None) if model else None
    if names is not None:
        return list(names)
    names = getattr(model, "feature_names_", None) if model else None
    if names is not None:
        return list(names)
    return DEFAULT_FEATURES

FEATURES = get_feature_order()

@app.route("/")
def index():
    return render_template("index.html",
                           features=FEATURES,
                           load_error=(None if model else load_err),
                           predict=None,
                           proba=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html",
                               features=FEATURES,
                               load_error=load_err,
                               predict=None,
                               proba=None), 500
    try:
        values = []
        for name in FEATURES:
            raw = request.form.get(name)
            if raw is None or raw.strip() == "":
                raise ValueError(f"Missing value for '{name}'")
            values.append(float(raw))
        X = np.array(values, dtype=float).reshape(1, -1)

        pred_class = int(model.predict(X)[0])

        target_names = getattr(model, "target_names_", ["malignant", "benign"])
        if len(target_names) != 2:
            target_names = ["malignant", "benign"]

        try:
            proba = float(model.predict_proba(X)[0, pred_class])
        except Exception:
            proba = None

        label = target_names[pred_class] if 0 <= pred_class < len(target_names) else str(pred_class)
        proba_pct = None if proba is None else f"{proba*100:.2f}%"

        return render_template("index.html",
                               features=FEATURES,
                               load_error=None,
                               predict=label.capitalize(),
                               proba=proba_pct)
    except Exception as e:
        return render_template("index.html",
                               features=FEATURES,
                               load_error=None,
                               predict=None,
                               proba=None,
                               form_error=str(e)), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

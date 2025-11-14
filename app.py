# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ---- Load model (pipeline) ----
# Expecting a pickle created from a Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(...))])
# and, ideally, with an attribute `feature_names_in_` (sklearn >=1.0) or a saved list of feature names.
MODEL_PATH = os.getenv("MODEL_PATH", "breast_lr.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    load_err = f"Could not load model from {MODEL_PATH}: {e}"

# Default feature names (30) if the pickle doesn't carry them.
DEFAULT_FEATURES = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

def get_feature_order():
    # Try sklearn's learned input feature names first
    names = getattr(model, "feature_names_in_", None) if model else None
    if names is not None:
        return list(names)
    # Try a custom attribute saved during training
    names = getattr(model, "feature_names_", None) if model else None
    if names is not None:
        return list(names)
    # Fall back to standard 30-feature order
    return DEFAULT_FEATURES

FEATURES = get_feature_order()

@app.route("/")
def index():
    # If model failed to load, show an error on the page
    return render_template("index.html",
                           features=FEATURES,
                           load_error=(None if model else load_err),
                           predict=None,
                           proba=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        # fail early if model wasn't loaded
        return render_template("index.html",
                               features=FEATURES,
                               load_error=load_err,
                               predict=None,
                               proba=None), 500
    try:
        # Collect inputs in the exact order the model expects
        values = []
        for name in FEATURES:
            raw = request.form.get(name)
            if raw is None or raw.strip() == "":
                raise ValueError(f"Missing value for '{name}'")
            values.append(float(raw))
        X = np.array(values, dtype=float).reshape(1, -1)

        pred_class = int(model.predict(X)[0])
        # For sklearn's breast cancer dataset: target names 0=malignant, 1=benign
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
        # Show a helpful message instead of a 500 page
        return render_template("index.html",
                               features=FEATURES,
                               load_error=None,
                               predict=None,
                               proba=None,
                               form_error=str(e)), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

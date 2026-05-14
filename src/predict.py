"""
src/predict.py
==============
Inference module — loads the pre-trained model and scaler from disk
and exposes a clean predict() function for use by the API and frontend.

WHY THIS FILE EXISTS:
  Separating inference from training is fundamental to MLOps. This module
  can be imported by any consumer (FastAPI, Streamlit, CLI) without knowing
  anything about how the model was trained.

INTERVIEW VALUE:
  Shows production-mindset: cached model loading, proper error messages,
  input validation, and a typed interface.
"""

import os
import functools
import numpy as np
import joblib
from typing import Dict, Any

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

# ─── Lazy-loaded singletons ──────────────────────────────────────────────────
# We use functools.lru_cache so the model and scaler are loaded once and then
# cached — this avoids expensive disk I/O on every API request.

@functools.lru_cache(maxsize=1)
def _load_artifacts() -> Dict[str, Any]:
    """
    Loads and caches the model and scaler from disk.

    Returns:
        Dict with keys 'model', 'model_name', 'scaler'.

    Raises:
        FileNotFoundError: If train.py has not been run yet.
    """
    for path, label in [(MODEL_PATH, "model.pkl"), (SCALER_PATH, "scaler.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{label}' not found at '{path}'.\n"
                "Run 'python src/train.py' to train and save the model first."
            )

    artifact = joblib.load(MODEL_PATH)
    artifact["scaler"] = joblib.load(SCALER_PATH)
    print(f"[Predict] Loaded model: {artifact['model_name']}")
    return artifact


def predict(features: list) -> Dict[str, Any]:
    """
    Runs inference on a single patient's feature vector.

    Workflow:
      1. Load (or retrieve cached) model + scaler.
      2. Validate input length against training feature count.
      3. Scale the raw input using the fitted scaler.
      4. Return prediction label and confidence probability.

    Args:
        features: List of raw (unscaled) feature values in the order defined
                  by get_feature_names() in preprocessing.py.

    Returns:
        {
          "prediction": 0 or 1,
          "label":      "Benign" or "Malignant",
          "confidence": float between 0 and 1,
          "model_used": str
        }

    Raises:
        ValueError: If features list has wrong length.
    """
    artifacts = _load_artifacts()
    model     = artifacts["model"]
    scaler    = artifacts["scaler"]
    model_name = artifacts["model_name"]

    # Validate input dimensionality
    expected_features = scaler.n_features_in_
    if len(features) != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, got {len(features)}. "
            "Ensure feature order matches training data."
        )

    # Scale input (same transformation applied during training)
    # Use DataFrame to preserve feature names and suppress sklearn warnings
    import pandas as pd
    feature_names = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None
    X_raw = np.array(features, dtype=float).reshape(1, -1)
    if feature_names:
        X_df = pd.DataFrame(X_raw, columns=feature_names)
        X_scaled = scaler.transform(X_df)
    else:
        X_scaled = scaler.transform(X_raw)

    # Predict
    prediction = int(model.predict(X_scaled)[0])
    confidence = float(model.predict_proba(X_scaled)[0][prediction])

    label = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": prediction,
        "label":      label,
        "confidence": round(confidence, 4),
        "model_used": model_name,
    }


def predict_batch(feature_matrix: list) -> list:
    """
    Runs inference on multiple samples at once.

    Args:
        feature_matrix: List of feature lists, shape (n_samples, n_features).

    Returns:
        List of result dicts (same structure as predict()).
    """
    return [predict(row) for row in feature_matrix]


def get_model_info() -> Dict[str, Any]:
    """
    Returns metadata about the currently loaded model.

    Useful for the health endpoint and the Streamlit metrics tab.
    """
    artifacts = _load_artifacts()
    model = artifacts["model"]
    return {
        "model_name":     artifacts["model_name"],
        "model_class":    type(model).__name__,
        "n_features":     artifacts["scaler"].n_features_in_,
        "model_path":     MODEL_PATH,
    }

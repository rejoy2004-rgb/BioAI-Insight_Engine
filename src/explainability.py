"""SHAP explanations for the fitted scikit-learn pipelines.

SHAP describes how the *model* arrives at its output, not biological causation.
Supported: Random Forest (TreeExplainer; contributions in score/probability
units) and Logistic Regression (LinearExplainer; contributions in log-odds).
Other models (SVM, baseline) are reported as unsupported rather than faked.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import FEATURES

log = logging.getLogger("bioai.explain")


class ExplanationUnavailable(RuntimeError):
    pass


def _clf(pipeline):
    return pipeline.named_steps["clf"]


def supports_shap(pipeline) -> bool:
    return isinstance(_clf(pipeline), (RandomForestClassifier, LogisticRegression))


def units_for(pipeline) -> str:
    return "log-odds" if isinstance(_clf(pipeline), LogisticRegression) else "model score (probability-like)"


def normalize_shap_values(values, n_features: int, positive_class: int = 1) -> np.ndarray:
    """Return a (n_samples, n_features) array for the positive class.

    Handles the shapes different SHAP versions/explainers produce:
      * list of per-class arrays                     [(n, f), (n, f)]
      * 3-D array (n, f, n_classes)
      * 2-D array (n, f)         (already single output)
      * 1-D array (f,)           (single sample)
    """
    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        elif positive_class < len(values):
            values = values[positive_class]
        else:
            raise ExplanationUnavailable("Unexpected SHAP output: class index out of range.")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:
        if arr.shape[1] != n_features:
            raise ExplanationUnavailable(f"Unexpected SHAP output shape {arr.shape}.")
        arr = arr[:, :, positive_class if arr.shape[2] > positive_class else 0]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ExplanationUnavailable(f"Unexpected SHAP output shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ExplanationUnavailable("SHAP produced non-finite values.")
    return arr


def normalize_base_value(expected, positive_class: int = 1) -> float:
    arr = np.atleast_1d(np.asarray(expected, dtype=float)).ravel()
    val = arr[positive_class] if arr.size > positive_class else arr[0]
    if not np.isfinite(val):
        raise ExplanationUnavailable("Non-finite SHAP base value.")
    return float(val)


class Explainer:
    """Wraps a fitted pipeline; construct once, reuse across requests."""

    def __init__(self, pipeline, background_raw: pd.DataFrame | None = None):
        import shap  # imported lazily so the API still serves predictions without SHAP
        if not supports_shap(pipeline):
            raise ExplanationUnavailable("SHAP is not supported for this model type.")
        self.pipeline = pipeline
        self.prep = pipeline[:-1]
        self.units = units_for(pipeline)
        clf = _clf(pipeline)
        try:
            if isinstance(clf, RandomForestClassifier):
                self._explainer = shap.TreeExplainer(clf)
            else:
                if background_raw is None:
                    raise ExplanationUnavailable("Linear SHAP needs a background sample.")
                bg = self.prep.transform(background_raw[FEATURES])
                self._explainer = shap.LinearExplainer(clf, bg)
            self.base_value = normalize_base_value(self._explainer.expected_value)
        except ExplanationUnavailable:
            raise
        except Exception as exc:  # shap raises many exception types across versions
            raise ExplanationUnavailable(f"Could not build SHAP explainer: {type(exc).__name__}") from exc

    def shap_values(self, X_raw: pd.DataFrame) -> np.ndarray:
        Xt = self.prep.transform(X_raw[FEATURES])
        try:
            raw = self._explainer.shap_values(Xt)
        except Exception as exc:
            raise ExplanationUnavailable(f"SHAP computation failed: {type(exc).__name__}") from exc
        return normalize_shap_values(raw, len(FEATURES))

    def explain_one(self, X_raw: pd.DataFrame) -> dict:
        sv = self.shap_values(X_raw.iloc[[0]])[0]
        row = X_raw.iloc[0]
        return {
            "method": "TreeExplainer" if isinstance(_clf(self.pipeline), RandomForestClassifier) else "LinearExplainer",
            "units": self.units,
            "base_value": self.base_value,
            "attributions": [{"feature": f, "value": float(row[f]), "shap_value": float(v)}
                             for f, v in zip(FEATURES, sv)],
        }


def compute_global_importance(pipeline, X_raw: pd.DataFrame, background_raw=None) -> dict:
    """Mean |SHAP| per feature over X_raw (training data), sorted descending."""
    ex = Explainer(pipeline, background_raw)
    mean_abs = np.abs(ex.shap_values(X_raw)).mean(axis=0)
    ordered = sorted(zip(FEATURES, mean_abs), key=lambda kv: -kv[1])
    return {"units": ex.units, "computed_on": "training partition",
            "importance": {f: float(v) for f, v in ordered}}

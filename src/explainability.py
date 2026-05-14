"""
src/explainability.py
=====================
Provides SHAP-based model explainability for the BioAI Insight Engine.

WHY THIS FILE EXISTS:
  "Black box" models are increasingly unacceptable in healthcare AI.
  Regulators, doctors, and patients need to understand WHY a model
  made a particular prediction. SHAP (SHapley Additive exPlanations)
  provides mathematically rigorous feature attributions.

INTERVIEW VALUE:
  XAI (Explainable AI) is a hot topic in ML interviews. Demonstrating
  that you understand SHAP values, tree explainers, and how to visualise
  feature contributions shows senior-level thinking.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def get_shap_explainer(model, X_train: np.ndarray):
    """
    Creates a SHAP explainer appropriate for the given model type.

    - TreeExplainer  → Random Forest, Gradient Boosting (fast, exact)
    - LinearExplainer → Logistic Regression (fast, exact)
    - KernelExplainer → any model (slow, model-agnostic fallback)

    Args:
        model:   A fitted scikit-learn estimator.
        X_train: Training data used to create background distribution for
                 KernelExplainer (only relevant for non-tree/linear models).

    Returns:
        A SHAP Explainer object.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                      "ExtraTreesClassifier", "DecisionTreeClassifier"):
        return shap.TreeExplainer(model)

    elif model_type == "LogisticRegression":
        # LinearExplainer needs a background dataset to compute expected value
        background = shap.maskers.Independent(X_train, max_samples=100)
        return shap.LinearExplainer(model, background)

    else:
        # Generic fallback — slower but works with any sklearn estimator
        background = shap.kmeans(X_train, 10)  # summarise background data
        return shap.KernelExplainer(model.predict_proba, background)


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_explain: np.ndarray,
):
    """
    Computes SHAP values for the given data.

    Args:
        model:     Fitted sklearn estimator.
        X_train:   Training data (for background distribution).
        X_explain: Data to explain (can be X_test or a single sample).

    Returns:
        shap_values: numpy array of shape (n_samples, n_features) for binary class 1,
                     or the raw SHAP output for multi-class.
        explainer:   The SHAP Explainer object (needed for base_values in waterfall plots).
    """
    import shap
    explainer = get_shap_explainer(model, X_train)
    shap_values = explainer.shap_values(X_explain)

    # For binary classification, TreeExplainer returns a list [class0, class1]
    # We want class 1 (Malignant) attributions
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    return shap_values, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    save: bool = True,
    title: str = "SHAP Summary Plot",
) -> plt.Figure:
    """
    Generates a SHAP beeswarm summary plot showing global feature importance.

    The summary plot shows:
      - Which features have the greatest impact (top = most important)
      - Whether high/low values push predictions toward malignant or benign

    Args:
        shap_values:   SHAP values array (n_samples, n_features).
        X_explain:     Feature data array (n_samples, n_features).
        feature_names: List of column names for axis labels.
        save:          If True, save to outputs/shap_summary.png.
        title:         Plot title.

    Returns:
        matplotlib Figure object (for embedding in Streamlit).
    """
    import shap
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        plot_type="dot",  # beeswarm
    )
    plt.title(title)
    plt.tight_layout()

    if save:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Explainability] SHAP summary saved → {out_path}")

    return plt.gcf()


def plot_shap_single_prediction(
    model,
    X_train: np.ndarray,
    sample: np.ndarray,
    feature_names: List[str],
    save: bool = True,
) -> plt.Figure:
    """
    Generates a SHAP waterfall (force) plot for a single prediction.

    This is the most useful explainability view for a clinician: it shows
    exactly which features pushed this patient's risk score up or down.

    Args:
        model:         Fitted sklearn estimator.
        X_train:       Training data for background.
        sample:        Single sample, shape (1, n_features) or (n_features,).
        feature_names: Feature column names.
        save:          Save to outputs/shap_single.png.

    Returns:
        matplotlib Figure.
    """
    import shap
    sample = np.array(sample).reshape(1, -1)
    shap_values, explainer = compute_shap_values(model, X_train, sample)

    # Force plot rendered as matplotlib (not JS) for Streamlit compatibility
    shap.initjs()  # only needed for notebook/JS plots, harmless otherwise

    # Use bar plot for single sample — cleaner than force plot in production
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_values[0]]
    ax.barh(feature_names, shap_values[0], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title("Feature Contributions for This Prediction\n(Red = pushes toward Malignant, Green = toward Benign)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUTS_DIR, "shap_single.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Explainability] SHAP single-prediction plot saved → {out_path}")

    return fig


def get_feature_importances_df(model, feature_names: List[str]):
    """
    Extracts feature importances from tree-based models as a sorted DataFrame.

    Falls back to SHAP mean absolute values for non-tree models (e.g. SVM).

    Args:
        model:         Fitted estimator.
        feature_names: Column names.

    Returns:
        pandas DataFrame with columns ['Feature', 'Importance'], sorted descending.
    """
    import pandas as pd
    model_type = type(model).__name__

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression — use absolute coefficient magnitude
        importances = np.abs(model.coef_[0])
    else:
        # Fallback — equal importance
        importances = np.ones(len(feature_names)) / len(feature_names)

    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)

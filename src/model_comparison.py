"""
src/model_comparison.py
=======================
Trains and compares Logistic Regression, SVM, and Random Forest on the
cancer dataset, generating a full evaluation report and saved plots.

WHY THIS FILE EXISTS:
  In production ML, you never deploy a model without comparing it against
  baselines. This module formalises that comparison, making it easy to
  add new algorithms or evaluation metrics without touching the training script.

INTERVIEW VALUE:
  Shows you can build a model-selection pipeline, understand multiple evaluation
  metrics (not just accuracy), and produce visualisations that stakeholders can
  actually interpret.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from typing import Dict, Any

# Project-level constants
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns a dictionary of {model_name: unfitted_estimator} candidates.

    Adding a new model to the comparison is as simple as adding one line here.
    All hyperparameters are set to sensible, reproducible defaults.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,       # prevents ConvergenceWarning on small datasets
            random_state=42,
            C=1.0,               # regularisation strength (inverse of lambda)
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,    # required to compute ROC-AUC via predict_proba
            random_state=42,
            C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,           # use all CPU cores
        ),
    }


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates a fitted model and returns a metrics dictionary.

    Metrics included:
      - Accuracy  : overall correct predictions
      - Precision : avoids false positives (important when false alarm is costly)
      - Recall    : avoids false negatives (critical in medical screening)
      - F1-Score  : harmonic mean of precision and recall
      - ROC-AUC   : area under the ROC curve (threshold-independent)

    Args:
        model:  A fitted scikit-learn estimator.
        X_test: Scaled test features.
        y_test: True labels.

    Returns:
        Dictionary of metric name → value (rounded to 4 d.p.)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),               4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob),                    4),
    }


def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train,
    y_test,
    save_plots: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Trains all candidate models, evaluates them, and optionally saves plots.

    Args:
        X_train, X_test: Scaled feature arrays from preprocessing.preprocess().
        y_train, y_test: Label arrays.
        save_plots:      If True, saves ROC curve and confusion matrices to outputs/.

    Returns:
        Nested dict: {model_name: {metric: value}}
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    models = get_candidate_models()
    results: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Any] = {}

    for name, model in models.items():
        print(f"[ModelComparison] Training: {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(f"  → {metrics}")
        print(f"\n  Classification Report — {name}:")
        print(classification_report(y_test, model.predict(X_test)))

    if save_plots:
        _save_roc_curves(fitted_models, X_test, y_test)
        _save_confusion_matrices(fitted_models, X_test, y_test)
        _save_metrics_bar_chart(results)

    return results, fitted_models


def _save_roc_curves(
    fitted_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test,
) -> None:
    """
    Plots all models' ROC curves on the same axes and saves to outputs/roc_curves.png.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")

    for name, model in fitted_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUTPUTS_DIR, "roc_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ModelComparison] ROC curves saved → {out_path}")


def _save_confusion_matrices(
    fitted_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test,
) -> None:
    """
    Generates one confusion matrix per model and saves each as a PNG.
    """
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign (0)", "Malignant (1)"],
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")
        safe_name = name.lower().replace(" ", "_")
        out_path = os.path.join(OUTPUTS_DIR, f"cm_{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[ModelComparison] Confusion matrix saved → {out_path}")


def _save_metrics_bar_chart(results: Dict[str, Dict[str, float]]) -> None:
    """
    Side-by-side bar chart of all metrics across all models.
    """
    df = pd.DataFrame(results).T  # rows = models, cols = metrics
    ax = df.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Model Comparison — All Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    ax.get_figure().savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ModelComparison] Metrics bar chart saved → {out_path}")


def get_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """
    Returns the name of the model with the highest ROC-AUC score.

    ROC-AUC is preferred over accuracy because the cancer dataset may have
    class imbalance, and AUC measures discriminative ability regardless of threshold.
    """
    return max(results, key=lambda name: results[name]["ROC-AUC"])

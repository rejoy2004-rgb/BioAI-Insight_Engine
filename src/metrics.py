"""Evaluation metrics that return None (not 0) when a metric is undefined."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_div(num: float, den: float):
    return float(num / den) if den else None


def validate_scores(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays of equal length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate on empty data.")
    if not np.isfinite(y_score).all():
        raise ValueError("Scores contain NaN or infinity.")
    if (y_score < 0).any() or (y_score > 1).any():
        raise ValueError("Scores must be probabilities-like values in [0, 1].")
    return y_true, y_score


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_score[m].mean())
    return float(ece)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Binary metrics with malignant (1) as the positive class.

    Undefined quantities (e.g. recall with no positives, ROC-AUC with one class)
    are returned as None and a note is added.
    """
    y_true, y_score = validate_scores(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos, n_neg = int(tp + fn), int(tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    two_classes = n_pos > 0 and n_neg > 0
    notes = []
    if not two_classes:
        notes.append("Only one class present in y_true; ROC-AUC, PR-AUC and calibration are undefined.")
    return {
        "threshold": threshold,
        "n": int(len(y_true)), "n_positive": n_pos, "n_negative": n_neg,
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if two_classes else None,
        "brier": float(brier_score_loss(y_true, y_score)) if two_classes else None,
        "ece": expected_calibration_error(y_true, y_score) if two_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "notes": notes,
    }


def curves(y_true, y_score, max_points: int = 200) -> dict | None:
    """ROC and precision-recall curve points; None if undefined."""
    y_true, y_score = validate_scores(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    def thin(*arrs):
        step = max(1, len(arrs[0]) // max_points)
        return [np.round(a[::step], 4).tolist() for a in arrs]

    f, t = thin(fpr, tpr)
    p, r = thin(prec, rec)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"precision": p, "recall": r}}


def threshold_table(y_true, y_score, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_score, t)
        rows.append({k: m[k] for k in ("threshold", "precision", "recall_sensitivity", "specificity", "f1")}
                    | {"fn": m["confusion_matrix"]["fn"], "fp": m["confusion_matrix"]["fp"]})
    return rows

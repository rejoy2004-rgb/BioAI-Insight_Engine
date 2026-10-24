import numpy as np
import pytest

from src.metrics import compute_metrics, curves, expected_calibration_error, threshold_table


def test_perfect_classifier():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == m["precision"] == m["recall_sensitivity"] == m["specificity"] == m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_one_class_gives_undefined_not_zero():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.7])
    assert m["roc_auc"] is None and m["pr_auc"] is None and m["recall_sensitivity"] is None
    assert m["notes"]
    assert curves([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_no_predicted_positives_precision_undefined():
    m = compute_metrics([0, 1, 1], [0.1, 0.2, 0.3])
    assert m["precision"] is None and m["recall_sensitivity"] == 0.0


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [np.inf, 0.5], [-0.1, 0.5], [1.2, 0.5]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        compute_metrics([0, 1], scores)


def test_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        compute_metrics([0, 1, 1], [0.1, 0.9])
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_threshold_changes_tradeoff():
    y, s = [0, 0, 1, 1, 1], [0.2, 0.6, 0.4, 0.7, 0.9]
    lo, hi = compute_metrics(y, s, 0.3), compute_metrics(y, s, 0.8)
    assert lo["recall_sensitivity"] > hi["recall_sensitivity"]
    assert lo["specificity"] <= hi["specificity"]
    assert len(threshold_table(y, s)) == 9


def test_ece_zero_when_calibrated_and_large_when_inverted():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert expected_calibration_error(y, np.array([0.0] * 4 + [1.0] * 4)) == 0.0
    assert expected_calibration_error(y, np.array([0.9] * 4 + [0.1] * 4)) > 0.5

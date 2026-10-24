import numpy as np
import pytest

from bioai_testutil import MALIGNANT_ROW, VALID_ROW
from src.explainability import (
    Explainer,
    ExplanationUnavailable,
    compute_global_importance,
    normalize_base_value,
    normalize_shap_values,
    supports_shap,
)
from src.preprocessing import FEATURES

F = len(FEATURES)


# ---- output-shape variations produced by different SHAP versions / explainers ----
def test_normalize_list_of_class_arrays():
    sv = [np.zeros((2, F)), np.ones((2, F))]
    assert normalize_shap_values(sv, F).tolist() == np.ones((2, F)).tolist()


def test_normalize_3d_array_n_f_classes():
    arr = np.zeros((3, F, 2))
    arr[:, :, 1] = 5
    assert (normalize_shap_values(arr, F) == 5).all()


def test_normalize_2d_and_1d():
    assert normalize_shap_values(np.ones((4, F)), F).shape == (4, F)
    assert normalize_shap_values(np.ones(F), F).shape == (1, F)


def test_normalize_single_element_list():
    assert normalize_shap_values([np.ones((1, F))], F).shape == (1, F)


@pytest.mark.parametrize("bad", [np.ones((2, F + 1)), np.ones((2, F + 1, 2)), np.full((1, F), np.nan), np.ones((1, 2, 3, 4))])
def test_normalize_rejects_bad_shapes_and_nonfinite(bad):
    with pytest.raises(ExplanationUnavailable):
        normalize_shap_values(bad, F)


def test_base_value_variants():
    assert normalize_base_value(0.3) == 0.3
    assert normalize_base_value(np.array([0.4, 0.6])) == 0.6
    assert normalize_base_value([0.2]) == 0.2
    with pytest.raises(ExplanationUnavailable):
        normalize_base_value(np.nan)


# ---- real fitted pipelines ----
def _frame(row):
    import pandas as pd
    return pd.DataFrame([row])[FEATURES].astype(float)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_local_explanation_is_additive_and_consistent(service, name):
    pipe = service.pipelines[name]
    bg = service.explainers[name] and None
    ex = service.explainers[name]
    out = ex.explain_one(_frame(MALIGNANT_ROW))
    assert [a["feature"] for a in out["attributions"]] == FEATURES
    assert [a["value"] for a in out["attributions"]] == [MALIGNANT_ROW[f] for f in FEATURES]
    total = out["base_value"] + sum(a["shap_value"] for a in out["attributions"])
    model_out = pipe.predict_proba(_frame(MALIGNANT_ROW))[0, 1]
    if name == "random_forest":                      # probability space: additivity must hold
        assert total == pytest.approx(model_out, abs=1e-6)
        assert out["units"].startswith("model score")
    else:                                            # log-odds space
        assert 1 / (1 + np.exp(-total)) == pytest.approx(model_out, abs=1e-6)
        assert out["units"] == "log-odds"
    assert bg is None


def test_malignant_input_pushes_toward_malignant(service):
    ex = service.explainers[service.primary]
    mal = sum(a["shap_value"] for a in ex.explain_one(_frame(MALIGNANT_ROW))["attributions"])
    ben = sum(a["shap_value"] for a in ex.explain_one(_frame(VALID_ROW))["attributions"])
    assert mal > ben


def test_svm_and_baseline_are_unsupported_not_faked(service):
    assert not supports_shap(service.pipelines["svm"]) and not supports_shap(service.pipelines["baseline"])
    with pytest.raises(ExplanationUnavailable):
        Explainer(service.pipelines["svm"])
    assert "svm" not in service.explainers


def test_global_importance_computed_not_hardcoded(trained):
    meta = trained[1]
    for name in ("logistic_regression", "random_forest"):
        imp = meta["global_importance"][name]["importance"]
        assert set(imp) == set(FEATURES)
        vals = list(imp.values())
        assert vals == sorted(vals, reverse=True) and all(v >= 0 for v in vals) and sum(vals) > 0


def test_linear_explainer_needs_background(service):
    with pytest.raises(ExplanationUnavailable, match="background"):
        Explainer(service.pipelines["logistic_regression"], None)


def test_service_explain_degrades_gracefully(service, monkeypatch):
    ex = service.explainers[service.primary]

    def boom(self, X):
        raise ExplanationUnavailable("simulated shap failure")
    monkeypatch.setattr(type(ex), "explain_one", boom)
    out = service.predict([VALID_ROW], explain=True)[0]
    assert out["explanation"]["status"] == "unavailable" and "simulated" in out["explanation"]["reason"]
    assert out["prediction"] in (0, 1)           # the prediction itself is unaffected


def test_compute_global_importance_rejects_unsupported(service):
    with pytest.raises(ExplanationUnavailable):
        compute_global_importance(service.pipelines["svm"], _frame(VALID_ROW))

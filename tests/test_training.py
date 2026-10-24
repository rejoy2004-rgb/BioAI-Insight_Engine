import json
import shutil

import joblib
import pytest

from bioai_testutil import MALIGNANT_ROW, VALID_ROW, make_dataset
from src.ml_model import holdout_split, sha256_file, train_pipeline
from src.preprocessing import FEATURES, DataValidationError
from src.service import ModelService


def test_split_is_grouped_and_stratified():
    df = make_dataset()
    tr, te = holdout_split(df, seed=1)
    assert not set(df.Id.iloc[tr]) & set(df.Id.iloc[te])
    assert abs(df.Class.iloc[tr].mean() - df.Class.iloc[te].mean()) < 0.1
    assert not set(tr) & set(te) and len(tr) + len(te) == len(df)


def test_training_is_reproducible(synthetic_csv, tmp_path):
    a = train_pipeline(seed=3, data_path=synthetic_csv, models_dir=tmp_path / "a", quick=True)
    b = train_pipeline(seed=3, data_path=synthetic_csv, models_dir=tmp_path / "b", quick=True)
    assert a["test_metrics"] == b["test_metrics"]
    assert a["primary_model"] == b["primary_model"]


def test_metadata_contents(trained):
    _, meta = trained
    assert meta["features"] == FEATURES
    assert {"baseline", "logistic_regression", "svm", "random_forest"} == set(meta["test_metrics"])
    assert meta["test_metrics"]["baseline"]["roc_auc"] == 0.5
    assert meta["primary_model"] in {"logistic_regression", "random_forest", "svm"}
    assert 0 < meta["operating_threshold"] <= 0.5
    assert "bundle_sha256" in meta


def test_scaler_inside_pipeline_fitted_on_training_partition_only(trained):
    out, meta = trained
    bundle = joblib.load(out / "model_bundle.joblib")
    scaler = bundle["pipelines"]["logistic_regression"].named_steps["prep"].named_steps["scale"]
    assert scaler.n_samples_seen_ == meta["split"]["n_train"] < len(make_dataset())


def test_model_beats_baseline_on_separable_synthetic_data(trained):
    assert trained[1]["test_metrics"]["logistic_regression"]["roc_auc"] > 0.9


def test_single_class_data_rejected(tmp_path):
    df = make_dataset()
    df["Class"] = 0
    p = tmp_path / "one.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        train_pipeline(data_path=p, models_dir=tmp_path / "m", quick=True)


def test_invalid_training_data_rejected(tmp_path):
    df = make_dataset()
    df.loc[0, "Cell.size"] = 99
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        train_pipeline(data_path=p, models_dir=tmp_path / "m", quick=True)


def test_missing_artifacts_reported_not_raised(tmp_path):
    svc = ModelService(tmp_path / "nothing")
    assert svc.load() is False and not svc.ready and "not found" in svc.load_error


def test_tampered_bundle_rejected(trained, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(trained[0], d)
    with open(d / "model_bundle.joblib", "ab") as f:
        f.write(b"tamper")
    svc = ModelService(d)
    assert svc.load() is False and "integrity" in svc.load_error


def test_feature_list_mismatch_rejected(trained, tmp_path):
    d = tmp_path / "copy2"
    shutil.copytree(trained[0], d)
    bundle = joblib.load(d / "model_bundle.joblib")
    bundle["features"] = list(reversed(bundle["features"]))
    joblib.dump(bundle, d / "model_bundle.joblib")
    meta = json.loads((d / "metadata.json").read_text())
    meta["bundle_sha256"] = sha256_file(d / "model_bundle.joblib")
    (d / "metadata.json").write_text(json.dumps(meta))
    svc = ModelService(d)
    assert svc.load() is False and "feature" in svc.load_error


def test_predict_validation(service):
    out = service.predict([VALID_ROW])
    assert 0 <= out[0]["malignancy_score"] <= 1 and out[0]["prediction"] in (0, 1)
    with pytest.raises(DataValidationError, match="Unexpected columns"):
        service.predict([{**VALID_ROW, "Extra": 1}])
    missing = dict(VALID_ROW)
    del missing["Mitoses"]
    with pytest.raises(DataValidationError, match="required"):
        service.predict([missing])
    with pytest.raises(DataValidationError):
        service.predict([{**VALID_ROW, "Mitoses": float("nan")}])
    with pytest.raises(DataValidationError):
        service.predict([])


def test_feature_order_independent_of_dict_order(service):
    shuffled = dict(reversed(list(VALID_ROW.items())))
    assert service.predict([VALID_ROW])[0]["malignancy_score"] == service.predict([shuffled])[0]["malignancy_score"]


def test_malignant_profile_scores_higher(service):
    assert service.predict([MALIGNANT_ROW])[0]["malignancy_score"] > service.predict([VALID_ROW])[0]["malignancy_score"]


def test_invalid_model_scores_are_rejected(service, monkeypatch):
    pipe = service.pipelines[service.primary]
    import numpy as np
    monkeypatch.setattr(type(pipe), "predict_proba", lambda self, X: np.full((len(X), 2), np.nan))
    with pytest.raises(RuntimeError, match="invalid scores"):
        service.predict([VALID_ROW])

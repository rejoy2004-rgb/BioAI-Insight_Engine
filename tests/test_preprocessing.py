import numpy as np
import pytest

from bioai_testutil import make_dataset
from src.preprocessing import (
    FEATURES,
    DataValidationError,
    build_preprocessor,
    data_quality_report,
    load_data,
    validate_dataframe,
)


def test_real_dataset_loads_and_is_not_modified():
    df = load_data()
    assert df.shape == (699, 11)           # nothing silently dropped
    assert df["Bare.nuclei"].isna().sum() == 16
    assert set(df["Class"].unique()) == {0, 1}


def test_quality_report_real_data():
    q = data_quality_report(load_data())
    assert q["class_counts"] == {"benign": 458, "malignant": 241}
    assert q["missing_by_feature"] == {"Bare.nuclei": 16}
    assert q["duplicate_feature_rows"] > 0 and q["constant_features"] == []


def test_missing_column_rejected():
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(make_dataset().drop(columns=["Mitoses"]))


def test_empty_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(make_dataset().iloc[0:0])


@pytest.mark.parametrize("bad", [0, 11, -3])
def test_out_of_range_rejected(bad):
    df = make_dataset()
    df.loc[3, "Cell.size"] = bad
    with pytest.raises(DataValidationError, match="must lie in"):
        validate_dataframe(df)


def test_infinity_rejected():
    df = make_dataset()
    df.loc[0, "Cell.size"] = np.inf
    with pytest.raises(DataValidationError, match="infinity"):
        validate_dataframe(df)


def test_non_numeric_rejected():
    df = make_dataset()
    df["Cell.size"] = df["Cell.size"].astype(str)
    with pytest.raises(DataValidationError, match="numeric"):
        validate_dataframe(df)


def test_unexpected_target_rejected():
    df = make_dataset()
    df.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="target"):
        validate_dataframe(df)


def test_constant_feature_detected():
    df = make_dataset()
    df["Mitoses"] = 1
    assert "Mitoses" in data_quality_report(df)["constant_features"]


def test_all_missing_column_changes_width_so_it_is_detectable():
    # An all-NaN column has no median, so the imputer drops it. Inference always
    # requires all nine features, and the bundle loader checks the feature list,
    # so this state cannot silently shift feature positions in production.
    X = make_dataset()[FEATURES].copy()
    X["Mitoses"] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert out.shape[1] == len(FEATURES) - 1


def test_preprocessor_fit_on_train_only():
    df = make_dataset()
    train, test = df.iloc[:150][FEATURES], df.iloc[150:][FEATURES]
    prep = build_preprocessor().fit(train)
    scaler = prep.named_steps["scale"]
    imputed_train = prep.named_steps["impute"].transform(train)
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean(axis=0))
    full_mean = build_preprocessor().fit(df[FEATURES]).named_steps["scale"].mean_
    assert not np.allclose(scaler.mean_, full_mean)   # test data did not influence the fit
    assert prep.transform(test).shape == test.shape

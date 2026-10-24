"""Data loading, validation and the (unfitted) preprocessing pipeline.

Learned preprocessing (imputation, scaling) lives inside a scikit-learn
Pipeline so it is fit on training folds only and reused unchanged at
inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_PATH

FEATURES = [
    "Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", "Epith.c.size",
    "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses",
]
TARGET = "Class"          # 0 = benign, 1 = malignant
ID_COLUMN = "Id"          # sample-code number; used only for grouped splitting
FEATURE_MIN, FEATURE_MAX = 1, 10
CLASS_LABELS = {0: "benign", 1: "malignant"}


class DataValidationError(ValueError):
    """Raised when a dataset or input frame violates the expected schema."""


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check schema, types and ranges. Missing feature values are allowed
    (they are imputed inside the pipeline); invalid values are rejected."""
    missing = [c for c in FEATURES + ([TARGET] if require_target else []) if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("Dataset is empty.")
    X = df[FEATURES]
    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise DataValidationError(f"Column '{col}' must be numeric.")
    values = X.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError("Feature values contain infinity.")
    finite = values[~np.isnan(values)]
    if finite.size and (finite.min() < FEATURE_MIN or finite.max() > FEATURE_MAX):
        raise DataValidationError(
            f"Feature values must lie in [{FEATURE_MIN}, {FEATURE_MAX}]; "
            f"found range [{finite.min()}, {finite.max()}]."
        )
    if require_target:
        bad = set(df[TARGET].dropna().unique()) - set(CLASS_LABELS)
        if bad or df[TARGET].isna().any():
            raise DataValidationError(f"Unexpected or missing target values: {sorted(map(str, bad))}")
    return df


def load_data(path=DATASET_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset. No rows are dropped here."""
    df = pd.read_csv(path, encoding="utf-8")
    return validate_dataframe(df)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Descriptive, non-destructive data-quality summary."""
    X = df[FEATURES]
    nunique = X.nunique(dropna=True)
    counts = df[TARGET].value_counts().sort_index()
    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "class_counts": {CLASS_LABELS[int(k)]: int(v) for k, v in counts.items()},
        "malignant_fraction": float((df[TARGET] == 1).mean()),
        "missing_by_feature": {c: int(v) for c, v in X.isna().sum().items() if v},
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df else None,
        "constant_features": [c for c in FEATURES if nunique[c] <= 1],
        "low_variety_features": [c for c in FEATURES if 1 < nunique[c] <= 2],
    }


def build_preprocessor() -> Pipeline:
    """Median imputation + standardisation. Unfitted: fit on training data only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

"""
src/preprocessing.py
====================
Handles all data loading and preprocessing for the BioAI Insight Engine.

WHY THIS FILE EXISTS:
  Separating preprocessing from model training is a core production ML practice.
  It allows reuse across training, evaluation, and inference without code duplication.

INTERVIEW VALUE:
  Demonstrates understanding of the ML pipeline, feature engineering, and
  the critical importance of preventing train/test data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple

# Path constants — relative to project root
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cancer.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads and performs initial cleaning of the cancer dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Cleaned DataFrame with Id dropped and missing rows removed.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please ensure cancer.csv is in the data/ directory."
        )

    data = pd.read_csv(filepath)

    # Drop the identifier column — it carries no predictive signal
    if "Id" in data.columns:
        data = data.drop(columns=["Id"])

    # Remove rows with any missing values
    initial_rows = len(data)
    data = data.dropna()
    dropped = initial_rows - len(data)
    if dropped > 0:
        print(f"[Preprocessing] Dropped {dropped} rows with missing values.")

    # Ensure all feature columns are numeric (some datasets encode as strings)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()

    return data


def get_features_and_target(
    data: pd.DataFrame, target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into features (X) and target (y).

    Args:
        data:       Cleaned DataFrame.
        target_col: Name of the label column.

    Returns:
        Tuple of (X, y).
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y


def get_feature_names(filepath: str = DATA_PATH) -> list:
    """
    Returns the list of feature column names without loading the full data.

    Useful for building prediction request schemas and UI sliders.
    """
    data = pd.read_csv(filepath, nrows=1)
    cols = [c for c in data.columns if c not in ("Id", "Class")]
    return cols


def preprocess(
    test_size: float = 0.20,
    random_state: int = 42,
    save_scaler: bool = True,
) -> Tuple:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Extract features and target
      3. Train/test split  ← FIXES the data leakage bug in the original code
      4. Fit StandardScaler ONLY on training data
      5. Optionally persist the scaler for inference

    WHY WE FIT THE SCALER ONLY ON X_TRAIN:
      If you fit the scaler on the full dataset before splitting, the test set
      statistics "leak" into the training process, inflating reported accuracy.
      This is one of the most common bugs in student ML projects.

    Args:
        test_size:    Fraction of data reserved for testing (default 20 %).
        random_state: Seed for reproducibility.
        save_scaler:  Whether to persist the fitted scaler to disk.

    Returns:
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    data = load_data()
    X_raw, y = get_features_and_target(data)
    feature_names = list(X_raw.columns)

    # --- CRITICAL FIX: split BEFORE scaling ---
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler ONLY on training data, then apply to both sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)      # ← transform (not fit_transform)

    print(f"[Preprocessing] Train samples : {len(X_train)}")
    print(f"[Preprocessing] Test  samples : {len(X_test)}")
    print(f"[Preprocessing] Features      : {feature_names}")

    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[Preprocessing] Scaler saved → {SCALER_PATH}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def load_scaler() -> StandardScaler:
    """
    Loads a previously-fitted scaler from disk.

    Used by predict.py and the API so they don't refit on production data.

    Raises:
        FileNotFoundError: If scaler.pkl has not been created by train.py yet.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'. "
            "Run 'python src/train.py' first to train and save the scaler."
        )
    return joblib.load(SCALER_PATH)

"""Dataset loading helpers.

The Wisconsin Breast Cancer Dataset (Diagnostic) is bundled with scikit-learn, so
experiments run offline and identically on every machine. Any other tabular dataset
can be supplied as a CSV file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_wbcd() -> tuple[pd.DataFrame, pd.Series]:
    """Return the WBCD features ``X`` (569 x 30) and binary target ``y``.

    The target uses scikit-learn's encoding: 0 = malignant, 1 = benign.
    """
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def load_csv(path: str | Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV file and split it into features and target.

    Parameters
    ----------
    path:
        Location of the CSV file.
    target:
        Name of the column that holds the class label.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If ``target`` is not a column of the file.
    ValueError
        If the file contains missing values or non-numeric feature columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(frame.columns)}")

    y = frame[target]
    X = frame.drop(columns=[target])

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Dataset contains missing values; impute or drop them first.")
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns are not supported: {non_numeric}")

    return X, y

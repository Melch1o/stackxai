"""Cross-validated comparison of the base learners and the stacking ensemble."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from stackxai.models import base_learners, build_stacking_classifier

METRICS = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate every base learner and the stacking ensemble with stratified k-fold CV.

    Returns a table with one row per model and the mean and standard deviation of
    accuracy, F1 and ROC AUC across folds. The same folds are used for all models, so
    the comparison is paired and fully determined by ``seed``.
    """
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    models = dict(base_learners(seed))
    models["stacking"] = build_stacking_classifier(seed)

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=splitter, scoring=METRICS)
        row = {"model": name}
        for metric in METRICS:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")

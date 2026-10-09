"""Model explanations: global (permutation importance) and local (SHAP, optional)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def explain_global(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.Series:
    """Rank features by permutation importance of a *fitted* model.

    Importance is the mean drop in accuracy when one feature is shuffled. The method is
    model-agnostic, so it works for the stacking ensemble as well as for any base learner.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring="accuracy"
    )
    importance = pd.Series(result.importances_mean, index=X.columns, name="importance")
    return importance.sort_values(ascending=False)


def explain_local(
    model: BaseEstimator,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    n_background: int = 25,
    nsamples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Return SHAP values for the positive class, one row per explained sample.

    Requires the optional ``shap`` dependency (``pip install stackxai[explain]``).
    KernelSHAP is used because it works with any model, including the stacking ensemble.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap installed
        raise ImportError(
            "explain_local needs the optional dependency 'shap'. "
            "Install it with: pip install stackxai[explain]"
        ) from exc

    background = shap.sample(X_background, min(n_background, len(X_background)), random_state=seed)

    def positive_proba(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=X_background.columns)
        return model.predict_proba(frame)[:, 1]

    explainer = shap.KernelExplainer(positive_proba, background)
    values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    return pd.DataFrame(np.asarray(values), columns=X_explain.columns, index=X_explain.index)

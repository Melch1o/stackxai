"""Base learners and the two-stage stacking ensemble used in the thesis pipeline."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from stackxai.fuzzy_knn import FuzzyKNN


def base_learners(seed: int = 42) -> dict[str, BaseEstimator]:
    """Return the six first-stage learners, each wrapped with scaling where needed."""
    return {
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)
        ),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "fuzzy_knn": make_pipeline(StandardScaler(), FuzzyKNN(n_neighbors=7)),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "neural_net": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=seed),
        ),
        "naive_bayes": GaussianNB(),
    }


def build_stacking_classifier(seed: int = 42, inner_cv: int = 5) -> StackingClassifier:
    """Build the two-stage stacking ensemble.

    The meta-classifier is trained on *out-of-fold* probabilities of the base learners
    (``inner_cv`` folds), which prevents the second stage from seeing predictions made
    on data the first stage was trained on.
    """
    if inner_cv < 2:
        raise ValueError("inner_cv must be at least 2.")
    return StackingClassifier(
        estimators=list(base_learners(seed).items()),
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=inner_cv,
        n_jobs=1,
    )

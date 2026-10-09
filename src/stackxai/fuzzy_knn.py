"""Fuzzy k-nearest-neighbour classifier (Keller, Gray & Givens, 1985).

scikit-learn has no fuzzy k-NN, so a small scikit-learn compatible implementation is
provided. Class memberships of the neighbours are weighted by inverse distance, which
yields smooth, probability-like outputs suitable for stacking.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class FuzzyKNN(ClassifierMixin, BaseEstimator):
    """Fuzzy k-NN classifier.

    Parameters
    ----------
    n_neighbors:
        Number of neighbours that vote for each query point.
    m:
        Fuzzifier (> 1). Larger values make the weighting of distant neighbours flatter.
    """

    def __init__(self, n_neighbors: int = 5, m: float = 2.0):
        self.n_neighbors = n_neighbors
        self.m = m

    def fit(self, X, y):
        if self.m <= 1:
            raise ValueError("m must be greater than 1.")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        X, y = validate_data(self, X, y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._memberships = np.eye(len(self.classes_))[y_idx]  # crisp labels as memberships
        k = min(self.n_neighbors, len(X))
        self._nn = NearestNeighbors(n_neighbors=k).fit(X)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "_nn", "_memberships"])
        X = check_array(X)
        distances, indices = self._nn.kneighbors(X)
        distances = np.maximum(distances, 1e-12)  # avoid division by zero for exact matches
        weights = distances ** (-2.0 / (self.m - 1.0))
        weights /= weights.sum(axis=1, keepdims=True)
        # (n_samples, k, 1) * (n_samples, k, n_classes) -> sum over neighbours
        return (weights[:, :, None] * self._memberships[indices]).sum(axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

import numpy as np
import pytest

from stackxai.fuzzy_knn import FuzzyKNN


def _two_blobs():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.2], [5.0, 5.0], [5.1, 5.1], [5.0, 5.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_predicts_clear_clusters():
    X, y = _two_blobs()
    model = FuzzyKNN(n_neighbors=3).fit(X, y)
    assert model.predict([[0.05, 0.05], [5.05, 5.05]]).tolist() == [0, 1]


def test_probabilities_are_valid():
    X, y = _two_blobs()
    proba = FuzzyKNN(n_neighbors=4).fit(X, y).predict_proba([[2.5, 2.5], [0.0, 0.0]])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_exact_match_does_not_divide_by_zero():
    X, y = _two_blobs()
    proba = FuzzyKNN(n_neighbors=3).fit(X, y).predict_proba(X[:1])
    assert np.isfinite(proba).all()
    assert proba[0, 0] > 0.99


def test_k_larger_than_training_set_is_clipped():
    X, y = _two_blobs()
    assert FuzzyKNN(n_neighbors=100).fit(X, y).predict(X).shape == (6,)


@pytest.mark.parametrize("params", [{"m": 1.0}, {"m": 0.5}, {"n_neighbors": 0}])
def test_invalid_parameters(params):
    X, y = _two_blobs()
    with pytest.raises(ValueError):
        FuzzyKNN(**params).fit(X, y)

import pytest
from sklearn.model_selection import train_test_split

from stackxai.explain import explain_global, explain_local
from stackxai.models import base_learners


@pytest.fixture(scope="module")
def fitted(wbcd):
    X, y = wbcd
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = base_learners(0)["logreg"].fit(X_tr, y_tr)
    return model, X_tr, X_te, y_te


def test_global_importance_is_sorted_and_complete(fitted):
    model, _, X_te, y_te = fitted
    importance = explain_global(model, X_te, y_te, n_repeats=3, seed=0)
    assert len(importance) == X_te.shape[1]
    assert importance.is_monotonic_decreasing
    assert importance.iloc[0] > 0


def test_local_shap_values_shape(fitted):
    pytest.importorskip("shap")
    model, X_tr, X_te, _ = fitted
    values = explain_local(model, X_tr, X_te.head(2), n_background=10, nsamples=60, seed=0)
    assert values.shape == (2, X_te.shape[1])
    assert list(values.columns) == list(X_te.columns)

import pytest

from stackxai.evaluate import compare_models


def test_compare_models_table(wbcd):
    X, y = wbcd
    table = compare_models(X, y, cv=3, seed=0)
    assert "stacking" in table.index
    assert len(table) == 7  # six base learners + stacking
    assert {"accuracy_mean", "f1_mean", "roc_auc_mean"} <= set(table.columns)
    assert (table["accuracy_mean"] > 0.85).all()


def test_compare_models_is_deterministic(wbcd):
    X, y = wbcd
    first = compare_models(X.head(200), y.head(200), cv=3, seed=7)
    second = compare_models(X.head(200), y.head(200), cv=3, seed=7)
    assert first.equals(second)


def test_invalid_cv(wbcd):
    X, y = wbcd
    with pytest.raises(ValueError):
        compare_models(X, y, cv=1)

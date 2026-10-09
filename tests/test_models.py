import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

from stackxai.models import base_learners, build_stacking_classifier

EXPECTED = {"svm_rbf", "logreg", "fuzzy_knn", "decision_tree", "neural_net", "naive_bayes"}


def test_six_base_learners():
    assert set(base_learners()) == EXPECTED


def test_stacking_uses_all_base_learners():
    model = build_stacking_classifier()
    assert isinstance(model, StackingClassifier)
    assert {name for name, _ in model.estimators} == EXPECTED


def test_invalid_inner_cv():
    with pytest.raises(ValueError):
        build_stacking_classifier(inner_cv=1)


def test_stacking_reaches_reasonable_accuracy(wbcd):
    X, y = wbcd
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = build_stacking_classifier(seed=0).fit(X_tr, y_tr)
    assert model.score(X_te, y_te) > 0.93


def test_training_is_reproducible(wbcd):
    X, y = wbcd
    a = build_stacking_classifier(seed=1).fit(X, y).predict_proba(X.head(20))
    b = build_stacking_classifier(seed=1).fit(X, y).predict_proba(X.head(20))
    assert (a == b).all()

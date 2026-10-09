import pandas as pd
import pytest

from stackxai.data import load_csv, load_wbcd


def test_wbcd_shape_and_labels(wbcd):
    X, y = wbcd
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_load_wbcd_returns_copies():
    X1, _ = load_wbcd()
    X1.iloc[0, 0] = -1
    X2, _ = load_wbcd()
    assert X2.iloc[0, 0] != -1


def test_load_csv_roundtrip(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]}).to_csv(path, index=False)
    X, y = load_csv(path, target="label")
    assert list(X.columns) == ["a", "b"]
    assert y.tolist() == [0, 1]


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv", target="label")


def test_load_csv_missing_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "label": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_csv(path, target="wrong")


def test_load_csv_rejects_missing_values(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1.0, None], "label": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_csv(path, target="label")


def test_load_csv_rejects_non_numeric_features(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": ["x", "y"], "label": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-numeric"):
        load_csv(path, target="label")

import json

import pytest

from stackxai.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "stackxai" in capsys.readouterr().out


def test_evaluate_writes_json(tmp_path, capsys):
    out = tmp_path / "out" / "results.json"
    assert main(["evaluate", "--cv", "3", "--seed", "0", "--output", str(out)]) == 0
    assert "stacking" in capsys.readouterr().out
    assert "stacking" in json.loads(out.read_text())


def test_explain_prints_top_features(capsys):
    assert main(["explain", "--top", "3", "--seed", "0"]) == 0
    assert "Test accuracy" in capsys.readouterr().out


def test_csv_requires_target():
    with pytest.raises(SystemExit):
        main(["evaluate", "--csv", "some.csv"])


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_evaluate_on_custom_csv(tmp_path, wbcd, capsys):
    X, y = wbcd
    path = tmp_path / "wbcd_small.csv"
    small = X.iloc[:120, :6].assign(label=y.iloc[:120])
    small.to_csv(path, index=False)
    assert main(["evaluate", "--csv", str(path), "--target", "label", "--cv", "3"]) == 0
    assert "stacking" in capsys.readouterr().out

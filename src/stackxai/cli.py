"""Command-line interface: ``stackxai evaluate`` and ``stackxai explain``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

from stackxai import __version__
from stackxai.data import load_csv, load_wbcd
from stackxai.evaluate import compare_models
from stackxai.explain import explain_global
from stackxai.models import build_stacking_classifier


def _load(args: argparse.Namespace):
    if args.csv:
        if not args.target:
            raise SystemExit("error: --target is required together with --csv")
        return load_csv(args.csv, args.target)
    return load_wbcd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stackxai", description="Stacking-ensemble evaluation with explainability."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("evaluate", "cross-validate base learners and the stacking ensemble"),
        ("explain", "train the stacking ensemble and rank features by importance"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--csv", help="path to a CSV file (default: bundled WBCD dataset)")
        p.add_argument("--target", help="target column name (required with --csv)")
        p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
        p.add_argument("--output", help="write the result as JSON to this file")
        if name == "evaluate":
            p.add_argument("--cv", type=int, default=5, help="number of CV folds (default: 5)")
        else:
            p.add_argument("--top", type=int, default=10, help="features to show (default: 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y = _load(args)

    if args.command == "evaluate":
        table = compare_models(X, y, cv=args.cv, seed=args.seed)
        print(table.round(4).to_string())
        payload = table.to_dict(orient="index")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=args.seed
        )
        model = build_stacking_classifier(args.seed).fit(X_train, y_train)
        importance = explain_global(model, X_test, y_test, seed=args.seed).head(args.top)
        print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
        print(importance.round(4).to_string())
        payload = importance.to_dict()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

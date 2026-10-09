# stackxai

[![CI](https://github.com/YOUR-USERNAME/stackxai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/stackxai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reproducible evaluation of a **two-stage stacking ensemble** with built-in
**explainability** for tabular medical data. It is a small, tested extract of the
research pipeline developed for a PhD thesis on hybrid interpretable diagnostic
decision-support systems (Wisconsin Breast Cancer Dataset).

## What it does

- Trains six base learners: SVM (RBF), logistic regression, **fuzzy k-NN** (own
  scikit-learn compatible implementation), decision tree, neural network and naive Bayes.
- Combines them with a logistic-regression meta-classifier trained on **out-of-fold**
  probabilities (stacking).
- Compares all models with stratified k-fold cross-validation using identical, seeded folds.
- Explains models globally (permutation importance) and per sample (SHAP, optional).

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/stackxai.git
cd stackxai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,explain]"
```

The `explain` extra installs SHAP and can be skipped if only global importance is needed.

## Usage

Command line:

```bash
stackxai evaluate --cv 5 --seed 42 --output results/evaluation.json
stackxai explain --top 10
stackxai evaluate --csv my_data.csv --target label     # any numeric tabular dataset
```

Python API:

```python
from stackxai import load_wbcd, build_stacking_classifier, explain_global

X, y = load_wbcd()
model = build_stacking_classifier(seed=42).fit(X, y)
print(explain_global(model, X, y).head())
```

## Development

```bash
ruff check . && ruff format --check .   # lint and formatting
pytest --cov                            # tests with coverage
```

Continuous integration (`.github/workflows/ci.yml`) runs lint, tests on Python 3.10-3.12
with a 90 % coverage gate, builds the package and, for tags such as `v0.1.0`, publishes a
GitHub release with the built artifacts.

## Project layout

```
src/stackxai/   data.py, fuzzy_knn.py, models.py, evaluate.py, explain.py, cli.py
tests/          pytest suite (data, models, evaluation, explanations, CLI)
.github/        CI/CD workflow, issue and pull request templates
```

## Reproducibility

All randomness is controlled by a single `--seed`. Dependencies are declared with minimum
versions in `pyproject.toml`; the same seed and library versions give identical results.

## Citation and license

Released under the [MIT License](LICENSE). Fuzzy k-NN follows Keller, Gray and Givens (1985).

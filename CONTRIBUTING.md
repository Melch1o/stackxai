# Contributing

1. Fork the repository and create a feature branch: `git switch -c feature/short-name`.
2. Install in editable mode: `pip install -e ".[dev,explain]"`.
3. Run `ruff check . && ruff format --check . && pytest` before every push.
4. Open a pull request against `main`. CI must be green before merging.

Commit messages follow the imperative style (`Add fuzzy k-NN`, `Fix CSV validation`).

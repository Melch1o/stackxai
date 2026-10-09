"""stackxai: reproducible stacking-ensemble evaluation with built-in explainability."""

from stackxai.data import load_csv, load_wbcd
from stackxai.evaluate import compare_models
from stackxai.explain import explain_global, explain_local
from stackxai.fuzzy_knn import FuzzyKNN
from stackxai.models import base_learners, build_stacking_classifier

__version__ = "0.1.0"

__all__ = [
    "FuzzyKNN",
    "__version__",
    "base_learners",
    "build_stacking_classifier",
    "compare_models",
    "explain_global",
    "explain_local",
    "load_csv",
    "load_wbcd",
]

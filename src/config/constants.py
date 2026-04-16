# src/config/constants.py
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
import numpy as np

# Minimum improvement over DummyClassifier to be considered "solved"
# This is dataset-relative, not absolute — defensible in a paper
IMPROVEMENT_MARGIN: dict[str, float] = {
    "accuracy":          0.10,   # must beat dummy by 10 percentage points
    "f1_macro":          0.15,   # harder metric, higher margin required
    "f1_weighted":       0.10,
    "roc_auc":           0.10,
    "balanced_accuracy": 0.10,
}

# Fallback absolute floor — even if dummy is weak, score must clear this minimum
ABSOLUTE_FLOOR: dict[str, float] = {
    "accuracy":          0.60,
    "f1_macro":          0.40,
    "f1_weighted":       0.50,
    "roc_auc":           0.60,
    "balanced_accuracy": 0.55,
}

METRIC_SCORER = {
    "accuracy":          make_scorer(accuracy_score),
    "f1_macro":          make_scorer(f1_score, average="macro"),
    "f1_weighted":       make_scorer(f1_score, average="weighted"),
    "roc_auc":           make_scorer(roc_auc_score, multi_class="ovr"),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}


def compute_dummy_score(X, y, metric: str, random_seed: int = 42) -> float:
    """Compute DummyClassifier CV score for a dataset — gives the baseline floor."""
    dummy = DummyClassifier(strategy="most_frequent", random_state=random_seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    scorer = METRIC_SCORER[metric]
    scores = cross_val_score(dummy, X, y, cv=cv, scoring=scorer)
    return float(np.mean(scores))


def is_solved(best_score: float, metric: str, dummy_score: float) -> bool:
    """
    Deterministic solved check.
    Passes only if BOTH conditions hold:
    1. Beats dummy by at least IMPROVEMENT_MARGIN[metric]
    2. Clears the absolute floor for that metric
    Uses epsilon tolerance to avoid floating point edge cases.
    """
    eps = 1e-9
    margin_ok = best_score >= dummy_score + IMPROVEMENT_MARGIN[metric] - eps
    floor_ok = best_score >= ABSOLUTE_FLOOR[metric] - eps
    return margin_ok and floor_ok
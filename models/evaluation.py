"""
evaluation.py – Hold-out metrics for the ensemble (accuracy, log loss, confusion matrix).
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split


def baseline_accuracy(y: np.ndarray) -> float:
    """Always predict the majority class."""
    values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def evaluate_ensemble_chronological(
    ensemble,
    X: np.ndarray,
    y: np.ndarray,
    sample_seasons: List[int],
    test_season: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate with temporal split: train on older seasons, test on latest season.
    """
    seasons_arr = np.array(sample_seasons)
    if test_season is None:
        test_season = int(max(sample_seasons))

    train_mask = seasons_arr != test_season
    test_mask = seasons_arr == test_season

    if test_mask.sum() < 10 or train_mask.sum() < 50:
        # Fallback: last 20% chronologically
        split_idx = int(len(X) * 0.8)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask
        test_season = int(max(sample_seasons))
        eval_mode = "chronological_80_20"
    else:
        eval_mode = f"season_holdout_{test_season}"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    ensemble.train(X_train, y_train)

    proba = ensemble.predict_proba_batch(X_test)
    y_pred = np.argmax(proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    base = baseline_accuracy(y_test)

    train_seasons = sorted(set(seasons_arr[train_mask].tolist()))
    test_seasons = sorted(set(seasons_arr[test_mask].tolist()))

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "baseline_accuracy": round(base, 4),
        "improvement_vs_baseline": round(acc - base, 4),
        "confusion_matrix": cm,
        "class_labels": ["HomeWin", "Draw", "AwayWin"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_scores": ensemble.cv_scores,
        "active_models": ensemble.active_model_names(),
        "eval_mode": eval_mode,
        "test_season": test_season,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
    }


def evaluate_ensemble(
    ensemble,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate ensemble on a stratified hold-out split.
    Expects ensemble already trained on full data or train split only.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Refit on train only for unbiased hold-out metrics
    ensemble.train(X_train, y_train)

    proba = ensemble.predict_proba_batch(X_test)
    y_pred = np.argmax(proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    base = baseline_accuracy(y_test)

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "baseline_accuracy": round(base, 4),
        "improvement_vs_baseline": round(acc - base, 4),
        "confusion_matrix": cm,
        "class_labels": ["HomeWin", "Draw", "AwayWin"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_scores": ensemble.cv_scores,
        "active_models": ensemble.active_model_names(),
    }


def load_metrics(path) -> Optional[Dict[str, Any]]:
    import json

    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

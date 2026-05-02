"""
tests/test_models.py – Unit tests for ML models and ensemble.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from data.preprocessor import Preprocessor

N = 200  # small dataset for fast tests


@pytest.fixture(scope="module")
def train_data():
    X, y = Preprocessor.generate_synthetic_training_data(n_samples=N, seed=0)
    p = Preprocessor()
    p.fit_scaler(X)
    X_scaled = p.scale(
        p.extract_features({
            "home_stats": {"win_rate": 0.5, "avg_goals_scored": 1.5,
                           "avg_goals_conceded": 1.5, "form_score": 1.5,
                           "wins": 5, "draws": 2, "losses": 3},
            "away_stats": {"win_rate": 0.4, "avg_goals_scored": 1.2,
                           "avg_goals_conceded": 1.8, "form_score": 1.2,
                           "wins": 4, "draws": 2, "losses": 4},
            "h2h": {"home_wins": 2, "draws": 1, "away_wins": 2, "total": 5}
        })
    )
    return X, y, X_scaled


def test_xgboost_proba_shape(train_data):
    from models.xgboost_model import XGBoostModel
    X, y, X_s = train_data
    m = XGBoostModel()
    m.train(X, y)
    proba = m.predict_proba(X_s)
    assert proba.shape == (1, 3)


def test_xgboost_proba_sums_to_one(train_data):
    from models.xgboost_model import XGBoostModel
    X, y, X_s = train_data
    m = XGBoostModel()
    m.train(X, y)
    proba = m.predict_proba(X_s)
    assert abs(proba[0].sum() - 1.0) < 1e-5


def test_random_forest_proba_shape(train_data):
    from models.random_forest_model import RandomForestModel
    X, y, X_s = train_data
    m = RandomForestModel()
    m.train(X, y)
    proba = m.predict_proba(X_s)
    assert proba.shape == (1, 3)


def test_ensemble_predict_keys(train_data):
    from models.ensemble import EnsembleModel
    X, y, X_s = train_data
    ens = EnsembleModel()
    ens.train(X, y)
    result = ens.predict(X_s)
    for key in ["probabilities", "outcome", "confidence", "model_breakdown", "agreement"]:
        assert key in result, f"Missing key: {key}"


def test_ensemble_probabilities_sum(train_data):
    from models.ensemble import EnsembleModel
    X, y, X_s = train_data
    ens = EnsembleModel()
    ens.train(X, y)
    result = ens.predict(X_s)
    total = sum(result["probabilities"].values())
    assert abs(total - 100.0) < 0.5, f"Probabilities sum to {total}, expected 100"

"""
tests/test_preprocessor.py – Unit tests for the feature engineering pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from data.preprocessor import Preprocessor, FEATURE_NAMES


SAMPLE_MATCH = {
    "home_stats": {
        "win_rate": 0.6, "avg_goals_scored": 2.1, "avg_goals_conceded": 1.2,
        "form_score": 2.0, "wins": 6, "draws": 2, "losses": 2,
    },
    "away_stats": {
        "win_rate": 0.4, "avg_goals_scored": 1.4, "avg_goals_conceded": 1.8,
        "form_score": 1.2, "wins": 4, "draws": 2, "losses": 4,
    },
    "h2h": {"home_wins": 3, "draws": 1, "away_wins": 1, "total": 5},
}


def test_feature_count():
    p = Preprocessor()
    df = p.extract_features(SAMPLE_MATCH)
    assert df.shape == (1, len(FEATURE_NAMES)), \
        f"Expected {len(FEATURE_NAMES)} features, got {df.shape[1]}"


def test_feature_names():
    p = Preprocessor()
    df = p.extract_features(SAMPLE_MATCH)
    assert list(df.columns) == FEATURE_NAMES


def test_elo_diff_positive_for_stronger_home():
    p = Preprocessor()
    df = p.extract_features(SAMPLE_MATCH)
    assert df["elo_diff"].iloc[0] > 0, "Stronger home team should have positive elo_diff"


def test_win_rate_diff():
    p = Preprocessor()
    df = p.extract_features(SAMPLE_MATCH)
    expected = 0.6 - 0.4
    assert abs(df["win_rate_diff"].iloc[0] - expected) < 1e-6


def test_no_nan_values():
    p = Preprocessor()
    df = p.extract_features(SAMPLE_MATCH)
    assert not df.isnull().any().any(), "Features should not contain NaN"


def test_synthetic_data_shape():
    X, y = Preprocessor.generate_synthetic_training_data(n_samples=100)
    assert X.shape == (100, len(FEATURE_NAMES))
    assert y.shape == (100,)
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_scale_output_shape():
    p = Preprocessor()
    X_raw, _ = Preprocessor.generate_synthetic_training_data(n_samples=50)
    p.fit_scaler(X_raw)
    df = p.extract_features(SAMPLE_MATCH)
    X_scaled = p.scale(df)
    assert X_scaled.shape == (1, len(FEATURE_NAMES))

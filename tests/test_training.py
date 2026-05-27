"""Tests for training pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from training.pipeline import generate_training_data, models_are_ready, train_models


def test_generate_training_data_shape():
    X, y = generate_training_data(n_samples=100)
    assert X.shape[0] == 100
    assert len(y) == 100


def test_train_models_creates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "ENSEMBLE_PATH", tmp_path / "ensemble.pkl")
    monkeypatch.setattr(config, "SCALER_PATH", tmp_path / "scaler.pkl")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")

    metrics = train_models(force=True, use_real=False, n_samples=150, evaluate=True)
    assert metrics["models_ready"] is True
    assert (tmp_path / "ensemble.pkl").exists()
    assert "accuracy" in metrics

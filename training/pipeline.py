"""
pipeline.py – Shared training / loading logic for CLI and Streamlit.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

import config
from data.preprocessor import Preprocessor
from models.ensemble import EnsembleModel
from models.evaluation import (
    evaluate_ensemble,
    evaluate_ensemble_chronological,
    load_metrics as _load_metrics_file,
)
from training.real_dataset import build_real_dataset


def generate_training_data(n_samples: Optional[int] = None):
    n = n_samples or config.TRAINING_SAMPLES
    return Preprocessor.generate_synthetic_training_data(
        n_samples=n,
        seed=config.TRAINING_RANDOM_STATE,
    )


def models_are_ready() -> bool:
    return config.ENSEMBLE_PATH.exists() and config.SCALER_PATH.exists()


def load_metrics() -> Optional[Dict[str, Any]]:
    return _load_metrics_file(config.METRICS_PATH)


def train_models(
    force: bool = False,
    use_real: bool = True,
    n_samples: Optional[int] = None,
    evaluate: bool = True,
    competitions: Optional[List[str]] = None,
    seasons: Optional[List[int]] = None,
    refresh_data: bool = False,
) -> Dict[str, Any]:
    """
    Train ensemble, fit scaler, evaluate and persist metrics.
    By default uses real historical data from football-data.org.
    """
    dataset_meta: Dict[str, Any] = {}

    if use_real:
        print("[VictorIA] Construction du dataset réel (football-data.org) …")
        X, y, dataset_meta = build_real_dataset(
            competitions=competitions,
            seasons=seasons,
            force_refresh=force or refresh_data,
        )
        data_source = "football-data.org-real"
        sample_seasons = dataset_meta.get("seasons", [])
    else:
        print(
            f"[VictorIA] Génération synthétique "
            f"({n_samples or config.TRAINING_SAMPLES} échantillons) …"
        )
        X, y = generate_training_data(n_samples)
        data_source = "synthetic"
        sample_seasons = []

    preprocessor = Preprocessor()
    preprocessor.fit_scaler(X)

    result: Dict[str, Any] = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(X),
        "models_ready": True,
        "data_source": data_source,
        "competitions": dataset_meta.get("competitions_fetched", competitions),
        "seasons_requested": dataset_meta.get("seasons_requested", seasons),
        "seasons_used": dataset_meta.get("seasons_fetched", []),
        "total_matches_raw": dataset_meta.get("total_matches"),
        "label_distribution": dataset_meta.get("label_distribution"),
        "skipped_fetch": dataset_meta.get("skipped", []),
    }

    if evaluate:
        if use_real and sample_seasons:
            print("[VictorIA] Évaluation chronologique (saison la plus récente) …")
            eval_ensemble = EnsembleModel()
            holdout = evaluate_ensemble_chronological(
                eval_ensemble,
                X,
                y,
                sample_seasons,
            )
        else:
            print("[VictorIA] Évaluation hold-out aléatoire …")
            eval_ensemble = EnsembleModel()
            holdout = evaluate_ensemble(
                eval_ensemble,
                X,
                y,
                test_size=config.TRAINING_TEST_SIZE,
                random_state=config.TRAINING_RANDOM_STATE,
            )
        result.update(holdout)

    print("[VictorIA] Entraînement final sur tout le jeu …")
    ensemble = EnsembleModel()
    ensemble.train(X, y)
    result["cv_scores"] = ensemble.cv_scores
    result["active_models"] = ensemble.active_model_names()

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[VictorIA] Modèles sauvegardés → {config.ENSEMBLE_PATH}")
    print(f"[VictorIA] Métriques sauvegardées → {config.METRICS_PATH}")
    return result

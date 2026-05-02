"""
ensemble.py – Weighted soft-voting ensemble combining XGBoost, RF, and NN.
Automatically trains all sub-models if not already cached.
"""
import numpy as np
import joblib
from pathlib import Path
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from models.neural_net import NeuralNetModel

ENSEMBLE_CACHE = Path(__file__).parent.parent / "cache" / "ensemble.pkl"

# Model weights: [XGB, RF, NN]
DEFAULT_WEIGHTS = [0.40, 0.35, 0.25]
LABELS = ["Victoire Domicile", "Nul", "Victoire Extérieur"]
LABEL_SHORT = ["HomeWin", "Draw", "AwayWin"]


class EnsembleModel:
    """
    Weighted soft-voting ensemble.
    Exposes confidence scores and individual model breakdowns.
    """

    def __init__(self, weights: list[float] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.xgb = XGBoostModel()
        self.rf = RandomForestModel()
        self.nn = NeuralNetModel()
        self._trained = False

    # ──────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print("[Ensemble] Training sub-models …")
        self.xgb.train(X, y)
        self.rf.train(X, y)
        self.nn.train(X, y)
        self._trained = True
        self._save()
        print("[Ensemble] All sub-models trained and saved.")

    def _save(self):
        joblib.dump(self, ENSEMBLE_CACHE)

    @classmethod
    def load_or_train(cls, X: np.ndarray, y: np.ndarray,
                      force_retrain: bool = False) -> "EnsembleModel":
        if ENSEMBLE_CACHE.exists() and not force_retrain:
            print("[Ensemble] Loading from cache …")
            obj = joblib.load(ENSEMBLE_CACHE)
            return obj
        obj = cls()
        obj.train(X, y)
        return obj

    # ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> dict:
        """
        Returns a rich prediction dict:
          - probabilities: [p_home, p_draw, p_away]
          - outcome: predicted class label string
          - confidence: certainty score 0–100
          - model_breakdown: per-model probabilities
          - agreement: how much models agree (0–1)
        """
        p_xgb = self.xgb.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_nn = self.nn.predict_proba(X)

        w = np.array(self.weights)
        combined = (w[0] * p_xgb + w[1] * p_rf + w[2] * p_nn) / w.sum()
        combined = combined[0]  # single sample

        outcome_idx = int(np.argmax(combined))
        confidence = float(combined[outcome_idx]) * 100

        # Agreement: average pairwise cosine similarity
        all_preds = np.array([p_xgb[0], p_rf[0], p_nn[0]])
        agreement = self._agreement_score(all_preds)

        return {
            "probabilities": {
                "home_win": round(float(combined[0]) * 100, 1),
                "draw": round(float(combined[1]) * 100, 1),
                "away_win": round(float(combined[2]) * 100, 1),
            },
            "outcome": LABELS[outcome_idx],
            "outcome_key": LABEL_SHORT[outcome_idx],
            "confidence": round(confidence, 1),
            "model_breakdown": {
                "XGBoost": {k: round(v * 100, 1) for k, v in
                             zip(LABEL_SHORT, p_xgb[0])},
                "RandomForest": {k: round(v * 100, 1) for k, v in
                                  zip(LABEL_SHORT, p_rf[0])},
                "NeuralNet": {k: round(v * 100, 1) for k, v in
                               zip(LABEL_SHORT, p_nn[0])},
            },
            "agreement": round(agreement, 3),
            "raw_proba": combined,
        }

    @staticmethod
    def _agreement_score(preds: np.ndarray) -> float:
        """Average cosine similarity between model predictions."""
        from itertools import combinations
        sims = []
        for a, b in combinations(preds, 2):
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / norm) if norm > 0 else 0.0)
        return float(np.mean(sims))

    @property
    def cv_scores(self) -> dict:
        return {
            "XGBoost": self.xgb.cv_score,
            "RandomForest": self.rf.cv_score,
            "NeuralNet": self.nn.cv_score,
        }

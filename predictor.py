"""
predictor.py – Main prediction engine for VictorIA.
Orchestrates: data fetch → preprocess → ensemble predict → explain → report.
"""
import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from data.data_fetcher import DataFetcher
from data.preprocessor import Preprocessor
from models.ensemble import EnsembleModel
from explainability.shap_explainer import SHAPExplainer
from analysis.report_generator import ReportGenerator
from training.pipeline import models_are_ready, load_metrics, train_models

_OUTCOME_LABELS = ["Victoire Domicile", "Nul", "Victoire Extérieur"]
_OUTCOME_KEYS = ["HomeWin", "Draw", "AwayWin"]


class ModelsNotReadyError(FileNotFoundError):
    """Raised when cache models are missing."""


class MatchPredictor:
    """
    High-level API for predicting football match outcomes.

    Usage:
        predictor = MatchPredictor()
        report = predictor.predict("Paris SG", "Lyon", "Ligue 1")
    """

    def __init__(self, force_retrain: bool = False):
        self.fetcher = DataFetcher()
        self.preprocessor = Preprocessor()
        self.report_gen = ReportGenerator()
        self.metrics = load_metrics()

        if force_retrain:
            print("[VictorIA] Réentraînement demandé …")
            train_models(force=True)
        elif not models_are_ready():
            raise ModelsNotReadyError(
                "Les modèles ne sont pas entraînés. "
                "Exécutez d'abord : python train.py"
            )

        print("[VictorIA] Chargement des modèles …")
        self.ensemble = EnsembleModel.load()
        self.shap_exp = SHAPExplainer(
            self.ensemble.xgb,
            self.preprocessor.feature_names(),
        )
        print("[VictorIA] Prêt ✔")

    # ──────────────────────────────────────────────────────────
    def predict(
        self,
        home_team: str,
        away_team: str,
        competition: str = "Premier League",
    ) -> dict:
        """Full prediction pipeline. Returns the complete report dict."""
        print(f"\n[VictorIA] Analyse: {home_team} vs {away_team} ({competition})")

        match_data = self.fetcher.get_match_data(home_team, away_team, competition)
        feature_df = self.preprocessor.extract_features(match_data)
        X = self.preprocessor.scale(feature_df)

        prediction = self.ensemble.predict(X)
        if not match_data.get("synthetic"):
            prediction = self._blend_with_empirical(match_data, prediction)
        prediction["cv_scores"] = self.ensemble.cv_scores
        prediction["exact_score"] = self._predict_exact_score(match_data, prediction)

        outcome_map = {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
        outcome_idx = outcome_map.get(prediction["outcome_key"], 0)
        top_factors = self.shap_exp.get_top_factors(
            X, outcome_idx=outcome_idx, top_n=5
        )

        report = self.report_gen.generate(match_data, prediction, top_factors)
        report["feature_df"] = feature_df
        report["X_scaled"] = X
        report["outcome_idx"] = outcome_idx
        report["data_source"] = (
            "demo" if match_data.get("synthetic") else "api"
        )
        report["data_quality"] = match_data.get("data_quality", {})
        if self.metrics:
            report["model_metrics"] = self.metrics

        return report

    @staticmethod
    def _team_strength(stats: dict) -> float:
        goal_edge = stats["avg_goals_scored"] - stats["avg_goals_conceded"]
        return (
            0.45 * stats["win_rate"]
            + 0.35 * (stats["form_score"] / 3.0)
            + 0.20 * goal_edge
        )

    def _empirical_probabilities(self, match_data: dict) -> np.ndarray:
        """Derive 1X2 probabilities from real team stats + H2H."""
        hs = match_data["home_stats"]
        as_ = match_data["away_stats"]
        h2h = match_data["h2h"]

        home = self._team_strength(hs) + 0.05  # modest home edge
        away = self._team_strength(as_)
        gap = abs(home - away)
        draw = 0.24 + max(0.0, 0.12 - gap * 0.15)

        h2h_total = max(h2h.get("total", 0), 1)
        h2h_edge = (h2h.get("home_wins", 0) - h2h.get("away_wins", 0)) / h2h_total
        home += 0.08 * h2h_edge
        away -= 0.08 * h2h_edge

        logits = np.array([home, draw, away], dtype=float)
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def _blend_with_empirical(self, match_data: dict, prediction: dict) -> dict:
        """Correct ML bias when real API stats strongly disagree."""
        emp = self._empirical_probabilities(match_data)
        model = prediction["raw_proba"]
        w = config.EMPIRICAL_BLEND_WEIGHT
        blended = w * emp + (1.0 - w) * model
        blended = blended / blended.sum()

        outcome_idx = int(np.argmax(blended))
        confidence = float(blended[outcome_idx]) * 100

        prediction = dict(prediction)
        prediction["probabilities"] = {
            "home_win": round(float(blended[0]) * 100, 1),
            "draw": round(float(blended[1]) * 100, 1),
            "away_win": round(float(blended[2]) * 100, 1),
        }
        prediction["outcome"] = _OUTCOME_LABELS[outcome_idx]
        prediction["outcome_key"] = _OUTCOME_KEYS[outcome_idx]
        prediction["confidence"] = round(confidence, 1)
        prediction["raw_proba"] = blended
        prediction["empirical_adjusted"] = True
        return prediction

    @staticmethod
    def _predict_exact_score(match_data: dict, prediction: dict) -> dict:
        hs = match_data["home_stats"]
        as_ = match_data["away_stats"]
        probs = prediction["probabilities"]

        home_edge = (probs["home_win"] - probs["away_win"]) / 100.0
        base_xg_home = (
            0.55 * hs["avg_goals_scored"]
            + 0.45 * as_["avg_goals_conceded"]
            + 0.12
        )
        base_xg_away = (
            0.55 * as_["avg_goals_scored"]
            + 0.45 * hs["avg_goals_conceded"]
            - 0.08
        )

        xg_home = float(np.clip(base_xg_home * (1 + 0.18 * home_edge), 0.2, 4.5))
        xg_away = float(np.clip(base_xg_away * (1 - 0.18 * home_edge), 0.1, 4.0))

        def poisson_pmf(lmbda: float, k: int) -> float:
            return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

        score_matrix = []
        for hg in range(0, 6):
            for ag in range(0, 6):
                p = poisson_pmf(xg_home, hg) * poisson_pmf(xg_away, ag)
                score_matrix.append((hg, ag, p))

        score_matrix.sort(key=lambda x: x[2], reverse=True)
        top_scores = score_matrix[:5]

        outcome_key = prediction.get("outcome_key")
        if outcome_key == "HomeWin":
            consistent_scores = [s for s in score_matrix if s[0] > s[1]]
        elif outcome_key == "AwayWin":
            consistent_scores = [s for s in score_matrix if s[1] > s[0]]
        else:
            consistent_scores = [s for s in score_matrix if s[0] == s[1]]

        # The headline scoreline should match the predicted outcome.
        exact_source = consistent_scores[0] if consistent_scores else top_scores[0]
        exact_home = int(exact_source[0])
        exact_away = int(exact_source[1])

        return {
            "home_goals": exact_home,
            "away_goals": exact_away,
            "scoreline": f"{exact_home}-{exact_away}",
            "xg_home": round(xg_home, 2),
            "xg_away": round(xg_away, 2),
            "top_scorelines": [
                {
                    "scoreline": f"{hg}-{ag}",
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "probability_pct": round(prob * 100, 2),
                }
                for hg, ag, prob in top_scores
            ],
        }

    def get_shap_figure(self, report: dict):
        return self.shap_exp.plot_waterfall(
            report["X_scaled"],
            outcome_idx=report["outcome_idx"],
            top_n=10,
        )

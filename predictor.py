"""
predictor.py – Main prediction engine for VictorIA.
Orchestrates: data fetch → preprocess → ensemble predict → explain → report.
"""
import sys
import numpy as np
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from data.data_fetcher import DataFetcher
from data.preprocessor import Preprocessor
from models.ensemble import EnsembleModel
from explainability.shap_explainer import SHAPExplainer
from analysis.report_generator import ReportGenerator


def _generate_training_data():
    from data.preprocessor import Preprocessor
    X, y = Preprocessor.generate_synthetic_training_data(n_samples=5000)
    return X, y


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

        # Load or train ensemble
        print("[VictorIA] Initialisation du moteur de prédiction …")
        X, y = _generate_training_data()

        # Fit scaler on training data
        self.preprocessor.fit_scaler(X)

        self.ensemble = EnsembleModel.load_or_train(
            X, y, force_retrain=force_retrain
        )
        # SHAP explainer uses the XGBoost sub-model
        self.shap_exp = SHAPExplainer(
            self.ensemble.xgb,
            self.preprocessor.feature_names()
        )
        print("[VictorIA] Prêt ✔")

    # ──────────────────────────────────────────────────────────
    def predict(
        self,
        home_team: str,
        away_team: str,
        competition: str = "Premier League",
    ) -> dict:
        """
        Full prediction pipeline.
        Returns the complete report dict.
        """
        print(f"\n[VictorIA] Analyse: {home_team} vs {away_team} ({competition})")

        # 1. Fetch data
        match_data = self.fetcher.get_match_data(home_team, away_team, competition)

        # 2. Extract features
        feature_df = self.preprocessor.extract_features(match_data)
        X = self.preprocessor.scale(feature_df)

        # 3. Ensemble predict
        prediction = self.ensemble.predict(X)
        prediction["cv_scores"] = self.ensemble.cv_scores

        # 4. SHAP explanation
        outcome_map = {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
        outcome_idx = outcome_map.get(prediction["outcome_key"], 0)
        top_factors = self.shap_exp.get_top_factors(X, outcome_idx=outcome_idx,
                                                      top_n=5)

        # 5. Generate report
        report = self.report_gen.generate(match_data, prediction, top_factors)

        # Attach raw data for charts
        report["feature_df"] = feature_df
        report["X_scaled"] = X
        report["outcome_idx"] = outcome_idx

        return report

    def get_shap_figure(self, report: dict):
        """Generate SHAP waterfall figure from a prediction report."""
        return self.shap_exp.plot_waterfall(
            report["X_scaled"],
            outcome_idx=report["outcome_idx"],
            top_n=10,
        )

    def retrain(self):
        """Force retrain all models (call when new data is available)."""
        X, y = _generate_training_data()
        self.preprocessor.fit_scaler(X)
        self.ensemble = EnsembleModel.load_or_train(X, y, force_retrain=True)
        self.shap_exp = SHAPExplainer(
            self.ensemble.xgb,
            self.preprocessor.feature_names()
        )
        print("[VictorIA] Modèles réentraînés ✔")

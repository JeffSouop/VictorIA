"""
preprocessor.py – Feature engineering for the ML models.
Converts raw match data (team stats + H2H) into a normalized feature vector.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import config

SCALER_PATH = config.SCALER_PATH

FEATURE_NAMES = [
    # Home team features
    "home_win_rate",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_goal_diff_avg",
    "home_form_score",
    "home_wins",
    "home_draws",
    "home_losses",

    # Away team features
    "away_win_rate",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_goal_diff_avg",
    "away_form_score",
    "away_wins",
    "away_draws",
    "away_losses",

    # Relative / comparative features
    "win_rate_diff",
    "goals_scored_diff",
    "goals_conceded_diff",
    "form_score_diff",
    "goal_diff_diff",

    # Elo-like rating difference (derived from win rate)
    "elo_diff",

    # Head-to-head features
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_total_games",

    # Context
    "home_advantage",    # always 1.0 for standard match
]


class Preprocessor:
    """
    Transforms raw match data dict into a Feature DataFrame.
    Supports fitting a StandardScaler on training data.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────
    def extract_features(self, match_data: dict) -> pd.DataFrame:
        """Returns a 1-row DataFrame with all features (raw, unscaled)."""
        hs = match_data["home_stats"]
        as_ = match_data["away_stats"]
        h2h = match_data["h2h"]

        h2h_total = max(h2h["total"], 1)

        features = {
            # Home
            "home_win_rate": hs["win_rate"],
            "home_avg_goals_scored": hs["avg_goals_scored"],
            "home_avg_goals_conceded": hs["avg_goals_conceded"],
            "home_goal_diff_avg": hs["avg_goals_scored"] - hs["avg_goals_conceded"],
            "home_form_score": hs["form_score"],
            "home_wins": hs["wins"],
            "home_draws": hs["draws"],
            "home_losses": hs["losses"],

            # Away
            "away_win_rate": as_["win_rate"],
            "away_avg_goals_scored": as_["avg_goals_scored"],
            "away_avg_goals_conceded": as_["avg_goals_conceded"],
            "away_goal_diff_avg": as_["avg_goals_scored"] - as_["avg_goals_conceded"],
            "away_form_score": as_["form_score"],
            "away_wins": as_["wins"],
            "away_draws": as_["draws"],
            "away_losses": as_["losses"],

            # Relative
            "win_rate_diff": hs["win_rate"] - as_["win_rate"],
            "goals_scored_diff": hs["avg_goals_scored"] - as_["avg_goals_scored"],
            "goals_conceded_diff": hs["avg_goals_conceded"] - as_["avg_goals_conceded"],
            "form_score_diff": hs["form_score"] - as_["form_score"],
            "goal_diff_diff": (
                (hs["avg_goals_scored"] - hs["avg_goals_conceded"]) -
                (as_["avg_goals_scored"] - as_["avg_goals_conceded"])
            ),

            # Elo proxy: 400 * log10(wr_home / wr_away)
            "elo_diff": self._elo_diff(hs["win_rate"], as_["win_rate"]),

            # H2H
            "h2h_home_win_rate": h2h["home_wins"] / h2h_total,
            "h2h_draw_rate": h2h["draws"] / h2h_total,
            "h2h_away_win_rate": h2h["away_wins"] / h2h_total,
            "h2h_total_games": h2h_total,

            # Context
            "home_advantage": 1.0,
        }

        return pd.DataFrame([features], columns=FEATURE_NAMES)

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        """Apply standard scaling. Fits scaler if not yet fitted."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(df.values)
            joblib.dump(self.scaler, SCALER_PATH)
            return X
        return self.scaler.transform(df.values)

    def fit_scaler(self, X: np.ndarray):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        joblib.dump(self.scaler, SCALER_PATH)

    def feature_names(self) -> list[str]:
        return FEATURE_NAMES

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _elo_diff(wr_home: float, wr_away: float) -> float:
        eps = 1e-6
        wr_home = np.clip(wr_home, eps, 1 - eps)
        wr_away = np.clip(wr_away, eps, 1 - eps)
        return 400 * np.log10(wr_home / wr_away)

    @staticmethod
    def generate_synthetic_training_data(
        n_samples: int = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic training data with realistic distributions.
        Labels: 0=HomeWin, 1=Draw, 2=AwayWin
        """
        if n_samples is None:
            n_samples = config.TRAINING_SAMPLES
        rng = np.random.RandomState(seed)
        prep = Preprocessor()
        X_rows = []
        y = []

        for _ in range(n_samples):
            h_wr = rng.uniform(0.2, 0.8)
            a_wr = rng.uniform(0.2, 0.8)
            h_gs = rng.uniform(0.8, 2.8)
            a_gs = rng.uniform(0.8, 2.5)
            h_gc = rng.uniform(0.5, 2.2)
            a_gc = rng.uniform(0.5, 2.2)
            h_form = rng.uniform(0.5, 2.5)
            a_form = rng.uniform(0.5, 2.5)

            match_data = {
                "home_stats": {
                    "win_rate": h_wr,
                    "avg_goals_scored": h_gs,
                    "avg_goals_conceded": h_gc,
                    "form_score": h_form,
                    "wins": int(10 * h_wr),
                    "draws": rng.randint(1, 3),
                    "losses": max(0, 10 - int(10 * h_wr) - 2),
                },
                "away_stats": {
                    "win_rate": a_wr,
                    "avg_goals_scored": a_gs,
                    "avg_goals_conceded": a_gc,
                    "form_score": a_form,
                    "wins": int(10 * a_wr),
                    "draws": rng.randint(1, 3),
                    "losses": max(0, 10 - int(10 * a_wr) - 2),
                },
                "h2h": {
                    "home_wins": rng.randint(0, 5),
                    "draws": rng.randint(0, 3),
                    "away_wins": rng.randint(0, 5),
                    "total": 8,
                }
            }

            df = prep.extract_features(match_data)
            X_rows.append(df.values[0])

            # Label generation: biased by elo_diff + home advantage
            prob_home_base = 0.46 + 0.04  # 4% home advantage
            elo = prep._elo_diff(h_wr, a_wr)
            prob_home = _sigmoid(elo / 200) * 0.55 + prob_home_base * 0.45
            prob_home = np.clip(prob_home, 0.15, 0.75)
            prob_draw = rng.uniform(0.18, 0.30)
            prob_away = max(0.05, 1.0 - prob_home - prob_draw)
            total = prob_home + prob_draw + prob_away
            probs = [prob_home / total, prob_draw / total, prob_away / total]
            label = rng.choice([0, 1, 2], p=probs)
            y.append(label)

        return np.array(X_rows, dtype=np.float32), np.array(y)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

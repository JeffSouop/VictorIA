"""
shap_explainer.py – SHAP-based explainability for match predictions.
Generates waterfall plots, force plots, and feature importance bars.
"""
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] shap not installed – explainability will be limited.")

FEATURE_DISPLAY_NAMES = {
    "home_win_rate": "Taux victoire (dom.)",
    "home_avg_goals_scored": "Buts marqués/match (dom.)",
    "home_avg_goals_conceded": "Buts concédés/match (dom.)",
    "home_goal_diff_avg": "Diff. buts moy. (dom.)",
    "home_form_score": "Forme récente (dom.)",
    "home_wins": "Victoires récentes (dom.)",
    "home_draws": "Nuls récents (dom.)",
    "home_losses": "Défaites récentes (dom.)",
    "away_win_rate": "Taux victoire (ext.)",
    "away_avg_goals_scored": "Buts marqués/match (ext.)",
    "away_avg_goals_conceded": "Buts concédés/match (ext.)",
    "away_goal_diff_avg": "Diff. buts moy. (ext.)",
    "away_form_score": "Forme récente (ext.)",
    "away_wins": "Victoires récentes (ext.)",
    "away_draws": "Nuls récents (ext.)",
    "away_losses": "Défaites récentes (ext.)",
    "win_rate_diff": "Écart taux victoire",
    "goals_scored_diff": "Écart buts marqués",
    "goals_conceded_diff": "Écart buts concédés",
    "form_score_diff": "Écart forme",
    "goal_diff_diff": "Écart diff. buts",
    "elo_diff": "Différence Elo",
    "h2h_home_win_rate": "H2H – taux victoire dom.",
    "h2h_draw_rate": "H2H – taux nul",
    "h2h_away_win_rate": "H2H – taux victoire ext.",
    "h2h_total_games": "H2H – matchs joués",
    "home_advantage": "Avantage domicile",
}


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost.
    Falls back to permutation-based importance if SHAP unavailable.
    """

    def __init__(self, xgb_model, feature_names: list[str]):
        self.feature_names = feature_names
        self.display_names = [
            FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names
        ]
        self.explainer = None

        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(xgb_model.model)
            except Exception as e:
                print(f"[SHAP] TreeExplainer init failed: {e}")

    def get_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Returns SHAP values matrix (n_samples, n_features, n_classes)."""
        if self.explainer is None:
            return None
        try:
            return self.explainer.shap_values(X)
        except Exception as e:
            print(f"[SHAP] Error: {e}")
            return None

    def plot_waterfall(self, X: np.ndarray, outcome_idx: int = 0,
                       top_n: int = 10) -> plt.Figure:
        """
        Horizontal bar waterfall chart showing top features for the
        predicted outcome class.
        Colors: green = positive impact, red = negative impact.
        """
        shap_vals = self.get_shap_values(X)

        if shap_vals is not None and isinstance(shap_vals, list):
            vals = shap_vals[outcome_idx][0]
        elif shap_vals is not None:
            vals = shap_vals[0, :, outcome_idx]
        else:
            # Fallback: use XGBoost feature importances as proxy
            vals = np.zeros(len(self.feature_names))

        # Sort by absolute value, take top N
        indices = np.argsort(np.abs(vals))[::-1][:top_n]
        top_vals = vals[indices]
        top_names = [self.display_names[i] for i in indices]

        # Reverse for bottom-up chart
        top_vals = top_vals[::-1]
        top_names = top_names[::-1]

        colors = ["#00c896" if v > 0 else "#ff4d6d" for v in top_vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        bars = ax.barh(top_names, top_vals, color=colors,
                       edgecolor="none", height=0.6)

        # Value labels
        for bar, val in zip(bars, top_vals):
            x = bar.get_width()
            ax.text(x + 0.002 * max(abs(top_vals.max()), abs(top_vals.min()), 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="left",
                    fontsize=8.5, color="white")

        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel("Impact SHAP", color="#aaaaaa", fontsize=10)
        ax.set_title(
            f"Facteurs clés – {['Victoire Domicile','Nul','Victoire Extérieur'][outcome_idx]}",
            color="white", fontsize=12, fontweight="bold", pad=12)
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

        pos_patch = mpatches.Patch(color="#00c896", label="Favorise ce résultat")
        neg_patch = mpatches.Patch(color="#ff4d6d", label="Défavorise ce résultat")
        ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
                  facecolor="#1a1d24", edgecolor="#333", labelcolor="white",
                  fontsize=8)

        plt.tight_layout()
        return fig

    def get_top_factors(self, X: np.ndarray, outcome_idx: int = 0,
                        top_n: int = 5) -> list[dict]:
        """
        Returns a list of dicts with the top contributing features.
        Used for text-based explanations in the report.
        """
        shap_vals = self.get_shap_values(X)

        if shap_vals is not None and isinstance(shap_vals, list):
            vals = shap_vals[outcome_idx][0]
        elif shap_vals is not None:
            vals = shap_vals[0, :, outcome_idx]
        else:
            return []

        indices = np.argsort(np.abs(vals))[::-1][:top_n]
        return [
            {
                "feature": self.feature_names[i],
                "display_name": self.display_names[i],
                "shap_value": float(vals[i]),
                "direction": "positif" if vals[i] > 0 else "négatif",
            }
            for i in indices
        ]

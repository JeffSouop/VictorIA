"""
report_generator.py – Generates a rich match analysis report as a dict
that can be rendered in Streamlit or exported as Markdown.
"""
from datetime import datetime


FORM_EMOJI = {3: "✅", 1: "🟡", 0: "❌"}
OUTCOME_EMOJI = {"HomeWin": "🏠", "Draw": "🤝", "AwayWin": "✈️"}
# Seuils volontairement prudents (pas de sur-confiance)
CONFIDENCE_LABELS = {
    (0, 50): ("Très incertain", "#ff6b6b"),
    (50, 62): ("Incertain", "#ffa94d"),
    (62, 74): ("Modéré", "#ffd43b"),
    (74, 85): ("Confiant", "#74c0fc"),
    (85, 101): ("Élevée", "#69db7c"),
}


def _confidence_label(conf: float) -> tuple[str, str]:
    for (lo, hi), (label, color) in CONFIDENCE_LABELS.items():
        if lo <= conf < hi:
            return label, color
    return "Très confiant", "#69db7c"


def _form_bar(form_last5: list[int]) -> str:
    """Convert [3,1,0,3,3] to emoji string."""
    return " ".join(FORM_EMOJI.get(s, "❓") for s in form_last5)


class ReportGenerator:
    """
    Generates a structured analysis report from prediction + match data.
    Output: a dict with all sections ready to be rendered.
    """

    def generate(
        self,
        match_data: dict,
        prediction: dict,
        top_factors: list[dict],
    ) -> dict:
        home = match_data["home_team"]
        away = match_data["away_team"]
        competition = match_data.get("competition", "—")
        is_synthetic = match_data.get("synthetic", False)
        hs = match_data["home_stats"]
        as_ = match_data["away_stats"]
        h2h = match_data["h2h"]

        conf = prediction["confidence"]
        conf_label, conf_color = _confidence_label(conf)

        report = {
            # ── Header ──────────────────────────────────────────────
            "match": f"{home} vs {away}",
            "competition": competition,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "synthetic": is_synthetic,

            # ── Prediction ──────────────────────────────────────────
            "outcome": prediction["outcome"],
            "outcome_key": prediction["outcome_key"],
            "outcome_emoji": OUTCOME_EMOJI.get(prediction["outcome_key"], ""),
            "probabilities": prediction["probabilities"],
            "confidence": conf,
            "confidence_label": conf_label,
            "confidence_color": conf_color,
            "agreement": prediction["agreement"],
            "exact_score": prediction.get("exact_score", {}),
            "live_stats": match_data.get("live_stats", {}),

            # ── Model breakdown ─────────────────────────────────────
            "model_breakdown": prediction["model_breakdown"],
            "cv_scores": prediction.get("cv_scores", {}),

            # ── Key factors (SHAP) ───────────────────────────────────
            "top_factors": top_factors,

            # ── Team stats ───────────────────────────────────────────
            "home_stats": {
                "team": home,
                "win_rate_pct": round(hs["win_rate"] * 100, 1),
                "avg_goals_scored": hs["avg_goals_scored"],
                "avg_goals_conceded": hs["avg_goals_conceded"],
                "goal_diff": round(hs["avg_goals_scored"] - hs["avg_goals_conceded"], 2),
                "form_score": hs["form_score"],
                "form_bar": _form_bar(hs.get("form_last5", [3, 1, 3, 0, 3])),
                "record": f"{hs['wins']}V / {hs['draws']}N / {hs['losses']}D",
            },
            "away_stats": {
                "team": away,
                "win_rate_pct": round(as_["win_rate"] * 100, 1),
                "avg_goals_scored": as_["avg_goals_scored"],
                "avg_goals_conceded": as_["avg_goals_conceded"],
                "goal_diff": round(as_["avg_goals_scored"] - as_["avg_goals_conceded"], 2),
                "form_score": as_["form_score"],
                "form_bar": _form_bar(as_.get("form_last5", [0, 1, 3, 1, 0])),
                "record": f"{as_['wins']}V / {as_['draws']}N / {as_['losses']}D",
            },

            # ── Head to head ─────────────────────────────────────────
            "h2h": {
                "home_wins": h2h["home_wins"],
                "draws": h2h["draws"],
                "away_wins": h2h["away_wins"],
                "total": h2h["total"],
                "home_win_pct": round(h2h["home_wins"] / max(h2h["total"], 1) * 100, 1),
                "draw_pct": round(h2h["draws"] / max(h2h["total"], 1) * 100, 1),
                "away_win_pct": round(h2h["away_wins"] / max(h2h["total"], 1) * 100, 1),
            },

            # ── Narrative summary ────────────────────────────────────
            "summary": self._narrative(home, away, prediction, hs, as_, h2h,
                                       top_factors, conf_label),
        }
        return report

    @staticmethod
    def _narrative(home, away, pred, hs, as_, h2h, factors, conf_label) -> str:
        """Generates a French-language textual summary of the analysis."""
        outcome = pred["outcome"]
        conf = pred["confidence"]
        probs = pred["probabilities"]
        wr_diff = round((hs["win_rate"] - as_["win_rate"]) * 100, 1)
        exact = pred.get("exact_score", {})

        direction = "légèrement" if abs(wr_diff) < 10 else ("nettement" if abs(wr_diff) > 20 else "")

        top1 = factors[0]["display_name"] if factors else "la forme récente"

        h2h_part = ""
        if h2h["total"] >= 3:
            if h2h["home_wins"] > h2h["away_wins"]:
                h2h_part = f" Historiquement, {home} domine les confrontations directes ({h2h['home_wins']} victoires sur {h2h['total']})."
            elif h2h["away_wins"] > h2h["home_wins"]:
                h2h_part = f" Historiquement, {away} domine les confrontations directes ({h2h['away_wins']} victoires sur {h2h['total']})."
            else:
                h2h_part = f" Les confrontations directes sont équilibrées ({h2h['total']} matchs joués)."

        return (
            f"**Analyse {conf_label.lower()}** — Le modèle prédit une **{outcome}** "
            f"avec {conf:.1f}% de confiance "
            f"(Domicile {probs['home_win']}% · Nul {probs['draw']}% · Extérieur {probs['away_win']}%). "
            f"{home} affiche {direction} un meilleur taux de victoire ({hs['win_rate']*100:.1f}% vs {as_['win_rate']*100:.1f}%). "
            f"Le score exact le plus probable est **{exact.get('scoreline', '1-1')}** "
            f"(xG {exact.get('xg_home', 1.5)} - {exact.get('xg_away', 1.2)}). "
            f"Le facteur le plus décisif est **{top1}**.{h2h_part}"
        )

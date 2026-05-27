"""
stats_builder.py – Compute team / H2H stats from match history entries.
Shared by live fetcher and historical dataset builder.
"""
from typing import Dict, List, Tuple

FORM_WEIGHTS = [0.05, 0.08, 0.10, 0.15, 0.62]


def match_entry_for_team(
    home_id: int,
    away_id: int,
    home_goals: int,
    away_goals: int,
    team_id: int,
) -> Dict[str, int]:
    """Single match result from one team's perspective."""
    is_home = team_id == home_id
    scored = home_goals if is_home else away_goals
    conceded = away_goals if is_home else home_goals
    if scored > conceded:
        points = 3
    elif scored == conceded:
        points = 1
    else:
        points = 0
    return {"scored": scored, "conceded": conceded, "points": points}


def compute_team_stats(entries: List[Dict[str, int]], window: int = 10) -> Dict:
    """Aggregate recent match entries into team stats dict."""
    recent = entries[-window:]
    wins = draws = losses = 0
    gf = ga = 0
    form_scores = []

    for entry in recent:
        gf += entry["scored"]
        ga += entry["conceded"]
        pts = entry["points"]
        form_scores.append(pts)
        if pts == 3:
            wins += 1
        elif pts == 1:
            draws += 1
        else:
            losses += 1

    n = max(len(recent), 1)
    last5 = form_scores[-5:]
    form_score = (
        sum(w * s for w, s in zip(FORM_WEIGHTS, last5))
        if last5
        else 1.5
    )

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored": gf,
        "goals_conceded": ga,
        "avg_goals_scored": gf / n,
        "avg_goals_conceded": ga / n,
        "win_rate": wins / n,
        "form_score": form_score,
        "form_last5": last5,
        "matches_played": n,
    }


def compute_h2h(h2h_results: List[Tuple[int, int]]) -> Dict:
    """
    h2h_results: list of (home_goals, away_goals) for the upcoming fixture's
    home/away orientation (before the current match).
    """
    home_wins = draws = away_wins = 0
    for hg, ag in h2h_results:
        if hg > ag:
            home_wins += 1
        elif hg < ag:
            away_wins += 1
        else:
            draws += 1
    total = len(h2h_results)
    return {
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "total": total,
    }


def match_outcome_label(home_goals: int, away_goals: int) -> int:
    """0=HomeWin, 1=Draw, 2=AwayWin."""
    if home_goals > away_goals:
        return 0
    if home_goals < away_goals:
        return 2
    return 1

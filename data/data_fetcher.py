"""
data_fetcher.py – Fetches team statistics from public APIs.
Supports: football-data.org (real data) + synthetic demo mode.
"""
import os
import json
from typing import Optional
from difflib import get_close_matches
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FOOTBALL_API_BASE = "https://api.football-data.org/v4"
SPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

COMPETITION_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Ligue 1": "FL1",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Champions League": "CL",
}

TEAM_ALIASES = {
    "real": "Real Madrid CF",
    "real madrid": "Real Madrid CF",
    "madrid": "Real Madrid CF",
    "barca": "FC Barcelona",
    "barça": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "osasuna": "CA Osasuna",
    "getafe": "Getafe CF",
    "atleti": "Club Atlético de Madrid",
    "atletico": "Club Atlético de Madrid",
    "atletico madrid": "Club Atlético de Madrid",
    "psg": "Paris Saint-Germain FC",
    "paris": "Paris Saint-Germain FC",
    "man city": "Manchester City FC",
    "city": "Manchester City FC",
    "man united": "Manchester United FC",
    "man utd": "Manchester United FC",
    "united": "Manchester United FC",
    "spurs": "Tottenham Hotspur FC",
}


def _cache_key(url: str, params: dict) -> str:
    raw = url + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _cached_get(url: str, params: dict = None, headers: dict = None, ttl_hours: int = 6):
    """HTTP GET with disk cache."""
    params = params or {}
    key = _cache_key(url, params)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        age = (time.time() - cache_file.stat().st_mtime) / 3600
        if age < ttl_hours:
            with open(cache_file) as f:
                return json.load(f)

    try:
        resp = requests.get(url, params=params, headers=headers or {}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"[DataFetcher] API error: {e}")
        return None


class DataFetcher:
    """
    Fetches team statistics and match history.
    Falls back to synthetic data generation if no API key is present.
    """

    def __init__(self):
        self.api_key = os.getenv("FOOTBALL_API_KEY", "")
        self.use_real_data = bool(self.api_key)
        if self.use_real_data:
            print("[DataFetcher] Using football-data.org API")
        else:
            print("[DataFetcher] No API key – using synthetic demo data")

    # ──────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────
    def get_match_data(self, home_team: str, away_team: str,
                       competition: str = "Premier League") -> dict:
        """
        Returns a dict with all data needed by the preprocessor:
          - home_stats, away_stats  (recent form)
          - h2h                     (head to head history)
          - competition_context
        """
        if self.use_real_data:
            match_data = self._fetch_real(home_team, away_team, competition)
        else:
            match_data = self._generate_synthetic(home_team, away_team, competition)

        fetched_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        source = match_data.get("data_quality", {}).get(
            "source",
            "football-data.org" if self.use_real_data else "synthetic-demo",
        )
        match_data["live_stats"] = self._build_live_stats(match_data, fetched_at, source)
        return match_data

    # ──────────────────────────────────────────────────────────
    # Real API (football-data.org)
    # ──────────────────────────────────────────────────────────
    def _fetch_real(self, home_team, away_team, competition):
        headers = {"X-Auth-Token": self.api_key}
        # Search for teams
        home_match = self._find_team(home_team, headers, competition)
        away_match = self._find_team(away_team, headers, competition)

        if not home_match or not away_match:
            missing = []
            if not home_match:
                missing.append(home_team)
            if not away_match:
                missing.append(away_team)
            print(f"[DataFetcher] Équipe introuvable via API: {', '.join(missing)}")
            fallback = self._generate_synthetic(home_team, away_team, competition)
            fallback["data_quality"] = {
                "source": "synthetic-fallback",
                "warning": "Une ou plusieurs équipes n'ont pas été trouvées dans l'API.",
                "missing_teams": missing,
            }
            return fallback

        home_id = home_match["id"]
        away_id = away_match["id"]

        home_stats = self._team_stats(home_id, headers)
        away_stats = self._team_stats(away_id, headers)
        h2h = self._h2h(home_id, away_id, headers)

        return {
            "home_team": home_match["name"],
            "away_team": away_match["name"],
            "competition": competition,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "h2h": h2h,
            "synthetic": False,
            "data_quality": {
                "source": "football-data.org",
                "home_query": home_team,
                "away_query": away_team,
                "home_resolved": home_match["name"],
                "away_resolved": away_match["name"],
            },
        }

    def _find_team_id(self, name: str, headers: dict) -> Optional[int]:
        team = self._find_team(name, headers, None)
        return team["id"] if team else None

    def _find_team(self, name: str, headers: dict, competition: Optional[str]) -> Optional[dict]:
        """Resolve user input to a football-data.org team using aliases + fuzzy match."""
        query = _normalize_team_name(name)
        canonical = TEAM_ALIASES.get(query, name)
        code = COMPETITION_CODES.get(competition or "")

        data = None
        if code:
            data = _cached_get(
                f"{FOOTBALL_API_BASE}/competitions/{code}/teams",
                headers=headers,
                ttl_hours=24,
            )
        if not data:
            data = _cached_get(f"{FOOTBALL_API_BASE}/teams", headers=headers, ttl_hours=24)

        teams = (data or {}).get("teams", [])
        if not teams:
            return None

        canonical_norm = _normalize_team_name(canonical)
        normalized = {
            _normalize_team_name(team.get("name", "")): team
            for team in teams
        }

        for candidate in (canonical_norm, query):
            if candidate in normalized:
                return normalized[candidate]

        # Also check short names / tla values, useful for inputs like "RMA".
        for team in teams:
            values = [
                team.get("name", ""),
                team.get("shortName", ""),
                team.get("tla", ""),
            ]
            if query in [_normalize_team_name(value) for value in values if value]:
                return team

        names = list(normalized.keys())
        match = get_close_matches(canonical_norm, names, n=1, cutoff=0.55)
        if match:
            return normalized[match[0]]

        return None

    def _team_stats(self, team_id: int, headers: dict) -> dict:
        if not team_id:
            return self._default_stats()
        data = _cached_get(f"{FOOTBALL_API_BASE}/teams/{team_id}/matches",
                           params={"status": "FINISHED", "limit": 10},
                           headers=headers)
        if not data:
            return self._default_stats()
        return self._parse_team_matches(data, team_id)

    def _parse_team_matches(self, data: dict, team_id: int) -> dict:
        matches = data.get("matches", [])[-10:]
        wins = draws = losses = gf = ga = 0
        form_scores = []
        for m in matches:
            h_id = m["homeTeam"]["id"]
            hg = m["score"]["fullTime"]["home"] or 0
            ag = m["score"]["fullTime"]["away"] or 0
            is_home = (h_id == team_id)
            scored = hg if is_home else ag
            conceded = ag if is_home else hg
            gf += scored
            ga += conceded
            if scored > conceded:
                wins += 1; form_scores.append(3)
            elif scored == conceded:
                draws += 1; form_scores.append(1)
            else:
                losses += 1; form_scores.append(0)
        n = max(len(matches), 1)
        return {
            "wins": wins, "draws": draws, "losses": losses,
            "goals_scored": gf, "goals_conceded": ga,
            "avg_goals_scored": gf / n, "avg_goals_conceded": ga / n,
            "win_rate": wins / n,
            "form_score": sum(w * s for w, s in zip(
                [0.05, 0.08, 0.10, 0.15, 0.62], reversed(form_scores[-5:]))) if form_scores else 1.5,
            "form_last5": form_scores[-5:],
            "matches_played": n,
        }

    def _h2h(self, home_id, away_id, headers) -> dict:
        if not home_id or not away_id:
            return self._default_h2h()
        data = _cached_get(
            f"{FOOTBALL_API_BASE}/teams/{home_id}/matches",
            params={"status": "FINISHED", "limit": 50},
            headers=headers
        )
        if not data:
            return self._default_h2h()

        home_wins = draws = away_wins = total = 0
        for match in data.get("matches", []):
            h_id = match["homeTeam"]["id"]
            a_id = match["awayTeam"]["id"]
            if {h_id, a_id} != {home_id, away_id}:
                continue

            hg = match["score"]["fullTime"]["home"]
            ag = match["score"]["fullTime"]["away"]
            if hg is None or ag is None:
                continue

            total += 1
            home_goals = hg if h_id == home_id else ag
            away_goals = ag if h_id == home_id else hg
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals < away_goals:
                away_wins += 1
            else:
                draws += 1

        if total == 0:
            return self._default_h2h()
        return {
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "total": total,
        }

    def _default_stats(self):
        return {
            "wins": 5, "draws": 2, "losses": 3,
            "goals_scored": 18, "goals_conceded": 14,
            "avg_goals_scored": 1.8, "avg_goals_conceded": 1.4,
            "win_rate": 0.5, "form_score": 1.5,
            "form_last5": [3, 1, 3, 0, 3],
            "matches_played": 10,
        }

    def _default_h2h(self):
        return {"home_wins": 0, "draws": 0, "away_wins": 0, "total": 0}

    def _build_live_stats(self, match_data: dict, fetched_at: str, source: str) -> dict:
        hs = match_data["home_stats"]
        as_ = match_data["away_stats"]
        h2h = match_data["h2h"]

        form_delta = round(hs["form_score"] - as_["form_score"], 3)
        attack_delta = round(hs["avg_goals_scored"] - as_["avg_goals_scored"], 3)
        defense_delta = round(as_["avg_goals_conceded"] - hs["avg_goals_conceded"], 3)
        h2h_total = max(h2h.get("total", 0), 1)
        h2h_home_edge = round((h2h.get("home_wins", 0) - h2h.get("away_wins", 0)) / h2h_total, 3)

        momentum = "home" if form_delta > 0.08 else ("away" if form_delta < -0.08 else "balanced")

        return {
            "source": source,
            "updated_at": fetched_at,
            "form_delta": form_delta,
            "attack_delta": attack_delta,
            "defense_delta": defense_delta,
            "h2h_home_edge": h2h_home_edge,
            "momentum": momentum,
        }

    # ──────────────────────────────────────────────────────────
    # Synthetic data generator (demo mode)
    # ──────────────────────────────────────────────────────────
    def _generate_synthetic(self, home_team: str, away_team: str,
                             competition: str) -> dict:
        """
        Generates realistic synthetic stats seeded by team name hash.
        Ensures the same team always gets the same stats (reproducible).
        """
        rng_home = np.random.RandomState(abs(hash(home_team)) % (2**31))
        rng_away = np.random.RandomState(abs(hash(away_team)) % (2**31))

        def gen_stats(rng, is_home_advantage):
            base_wr = rng.uniform(0.25, 0.75)
            wr = min(1.0, base_wr + (0.07 if is_home_advantage else 0))
            n = 10
            wins = int(n * wr)
            draws = rng.randint(1, max(2, n - wins - 1))
            losses = n - wins - draws
            avg_gs = rng.uniform(0.8, 2.5)
            avg_gc = rng.uniform(0.6, 2.2)
            form_raw = list(rng.choice([0, 1, 3], size=5,
                                        p=[1-wr-0.1, 0.1, wr]))
            form_score = sum(w * s for w, s in zip(
                [0.05, 0.08, 0.10, 0.15, 0.62], form_raw))
            return {
                "wins": wins, "draws": draws, "losses": losses,
                "goals_scored": int(avg_gs * n),
                "goals_conceded": int(avg_gc * n),
                "avg_goals_scored": round(avg_gs, 2),
                "avg_goals_conceded": round(avg_gc, 2),
                "win_rate": round(wr, 3),
                "form_score": round(form_score, 3),
                "form_last5": form_raw,
                "matches_played": n,
            }

        # H2H seeded on combined teams
        rng_h2h = np.random.RandomState(
            (abs(hash(home_team)) + abs(hash(away_team))) % (2**31))
        hw = rng_h2h.randint(0, 6)
        aw = rng_h2h.randint(0, 5)
        dr = rng_h2h.randint(0, 4)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "competition": competition,
            "home_stats": gen_stats(rng_home, is_home_advantage=True),
            "away_stats": gen_stats(rng_away, is_home_advantage=False),
            "h2h": {
                "home_wins": hw, "draws": dr, "away_wins": aw,
                "total": hw + dr + aw
            },
            "synthetic": True,
        }


def _normalize_team_name(value: str) -> str:
    value = value.lower().strip()
    replacements = {
        ".": "",
        "-": " ",
        "_": " ",
        " cf": "",
        " fc": "",
        " afc": "",
        " club de futbol": "",
        "football club": "",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return " ".join(value.split())

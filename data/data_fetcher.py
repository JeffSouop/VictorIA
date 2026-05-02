"""
data_fetcher.py – Fetches team statistics from public APIs.
Supports: football-data.org (real data) + synthetic demo mode.
"""
import os
import json
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
            return self._fetch_real(home_team, away_team, competition)
        else:
            return self._generate_synthetic(home_team, away_team, competition)

    # ──────────────────────────────────────────────────────────
    # Real API (football-data.org)
    # ──────────────────────────────────────────────────────────
    def _fetch_real(self, home_team, away_team, competition):
        headers = {"X-Auth-Token": self.api_key}
        # Search for teams
        home_id = self._find_team_id(home_team, headers)
        away_id = self._find_team_id(away_team, headers)

        home_stats = self._team_stats(home_id, headers)
        away_stats = self._team_stats(away_id, headers)
        h2h = self._h2h(home_id, away_id, headers)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "competition": competition,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "h2h": h2h,
        }

    def _find_team_id(self, name: str, headers: dict) -> int | None:
        data = _cached_get(f"{FOOTBALL_API_BASE}/teams",
                           params={"name": name}, headers=headers)
        if data and data.get("teams"):
            return data["teams"][0]["id"]
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
            "matches_played": n,
        }

    def _h2h(self, home_id, away_id, headers) -> dict:
        if not home_id or not away_id:
            return self._default_h2h()
        data = _cached_get(
            f"{FOOTBALL_API_BASE}/teams/{home_id}/matches",
            params={"status": "FINISHED", "limit": 5},
            headers=headers
        )
        return self._default_h2h()  # simplified

    def _default_stats(self):
        return {
            "wins": 5, "draws": 2, "losses": 3,
            "goals_scored": 18, "goals_conceded": 14,
            "avg_goals_scored": 1.8, "avg_goals_conceded": 1.4,
            "win_rate": 0.5, "form_score": 1.5, "matches_played": 10,
        }

    def _default_h2h(self):
        return {"home_wins": 2, "draws": 1, "away_wins": 2, "total": 5}

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

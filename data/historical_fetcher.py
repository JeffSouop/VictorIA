"""
historical_fetcher.py – Fetch finished matches from football-data.org by competition/season.
"""
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import config
from data.data_fetcher import FOOTBALL_API_BASE, _cached_get

load_dotenv()


class HistoricalFetcher:
    """Collect historical FINISHED matches for training."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FOOTBALL_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "FOOTBALL_API_KEY manquante. Ajoutez-la dans .env pour l'entraînement réel."
            )
        self.headers = {"X-Auth-Token": self.api_key}

    @staticmethod
    def default_seasons(n: int = None) -> List[int]:
        """Return starting years for the last N completed/recent seasons."""
        n = n or config.DEFAULT_TRAINING_SEASONS
        now = datetime.now()
        # Season 2024/25 starts in 2024
        current_start = now.year if now.month >= 7 else now.year - 1
        return list(range(current_start - n, current_start))

    def fetch_competition_season(
        self, code: str, season: int
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Fetch FINISHED matches for one competition season.
        Returns (matches, error_message).
        """
        url = f"{FOOTBALL_API_BASE}/competitions/{code}/matches"
        params = {"season": season, "status": "FINISHED"}

        try:
            data = _cached_get(
                url,
                params=params,
                headers=self.headers,
                ttl_hours=168,
            )
        except Exception as exc:
            return [], str(exc)

        if data is None:
            return [], "API request failed"

        raw = data.get("matches", [])
        normalized = []
        for match in raw:
            parsed = self._normalize_match(match, code, season)
            if parsed:
                normalized.append(parsed)

        if not normalized and raw:
            return [], "No valid finished scores"
        if not normalized:
            return [], "Season unavailable or empty (plan API limit?)"

        return normalized, None

    def fetch_all(
        self,
        competitions: Optional[List[str]] = None,
        seasons: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Fetch all historical matches for given competitions and seasons.
        Returns (matches, metadata).
        """
        competitions = competitions or config.TOP5_COMPETITION_CODES
        seasons = seasons or self.default_seasons()

        all_matches: List[Dict[str, Any]] = []
        fetched: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        total_requests = len(competitions) * len(seasons)
        done = 0

        for code in competitions:
            for season in seasons:
                done += 1
                print(
                    f"[HistoricalFetcher] {code} season {season} "
                    f"({done}/{total_requests}) …"
                )
                matches, err = self.fetch_competition_season(code, season)
                if err:
                    skipped.append(
                        {"competition": code, "season": season, "reason": err}
                    )
                    print(f"  → ignoré : {err}")
                else:
                    all_matches.extend(matches)
                    fetched.append(
                        {
                            "competition": code,
                            "season": season,
                            "matches": len(matches),
                        }
                    )
                    print(f"  → {len(matches)} matchs")

                if done < total_requests:
                    time.sleep(config.API_REQUEST_DELAY_SEC)

        all_matches.sort(key=lambda m: m["utcDate"])

        meta = {
            "data_source": "football-data.org",
            "competitions": competitions,
            "seasons_requested": seasons,
            "seasons_fetched": sorted({m["season"] for m in all_matches}),
            "competitions_fetched": sorted({m["competition"] for m in all_matches}),
            "total_matches": len(all_matches),
            "fetch_details": fetched,
            "skipped": skipped,
        }
        return all_matches, meta

    @staticmethod
    def _normalize_match(
        match: Dict[str, Any], competition_code: str, season: int
    ) -> Optional[Dict[str, Any]]:
        score = match.get("score", {}).get("fullTime", {})
        home_goals = score.get("home")
        away_goals = score.get("away")
        if home_goals is None or away_goals is None:
            return None

        home = match.get("homeTeam", {})
        away = match.get("awayTeam", {})
        if not home.get("id") or not away.get("id"):
            return None

        return {
            "id": match["id"],
            "utcDate": match["utcDate"],
            "season": season,
            "competition": competition_code,
            "homeTeam": {"id": home["id"], "name": home.get("name", "")},
            "awayTeam": {"id": away["id"], "name": away.get("name", "")},
            "homeGoals": int(home_goals),
            "awayGoals": int(away_goals),
        }

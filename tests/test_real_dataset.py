"""Tests for real dataset builder (no API calls)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stats_builder import (
    compute_h2h,
    compute_team_stats,
    match_entry_for_team,
    match_outcome_label,
)
from training.real_dataset import build_real_dataset


def _make_match(mid, date, season, home_id, away_id, hg, ag, comp="PL"):
    return {
        "id": mid,
        "utcDate": date,
        "season": season,
        "competition": comp,
        "homeTeam": {"id": home_id, "name": f"Team{home_id}"},
        "awayTeam": {"id": away_id, "name": f"Team{away_id}"},
        "homeGoals": hg,
        "awayGoals": ag,
    }


def test_match_outcome_label():
    assert match_outcome_label(2, 1) == 0
    assert match_outcome_label(1, 1) == 1
    assert match_outcome_label(0, 2) == 2


def test_compute_team_stats_from_entries():
    entries = [
        match_entry_for_team(1, 2, 2, 0, 1),
        match_entry_for_team(1, 3, 1, 1, 1),
        match_entry_for_team(1, 4, 0, 3, 1),
    ]
    stats = compute_team_stats(entries, window=3)
    assert stats["wins"] == 1
    assert stats["draws"] == 1
    assert stats["losses"] == 1
    assert stats["matches_played"] == 3


def test_compute_h2h():
    h2h = compute_h2h([(2, 0), (1, 1), (0, 1)])
    assert h2h["home_wins"] == 1
    assert h2h["draws"] == 1
    assert h2h["away_wins"] == 1
    assert h2h["total"] == 3


def test_build_dataset_from_mock_matches(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, "REAL_DATASET_PATH", tmp_path / "real.pkl")
    monkeypatch.setattr(config, "REAL_DATASET_META_PATH", tmp_path / "real_meta.json")
    monkeypatch.setattr(config, "MIN_TEAM_MATCHES", 2)

    matches = [
        _make_match(1, "2023-08-01T00:00:00Z", 2023, 10, 20, 1, 0),
        _make_match(2, "2023-08-08T00:00:00Z", 2023, 20, 30, 2, 2),
        _make_match(3, "2023-08-15T00:00:00Z", 2023, 10, 30, 0, 2),
        _make_match(4, "2023-08-22T00:00:00Z", 2023, 30, 10, 1, 3),
        _make_match(5, "2024-08-01T00:00:00Z", 2024, 10, 20, 2, 1),
    ]

    class MockFetcher:
        def fetch_all(self, competitions=None, seasons=None):
            return matches, {
                "data_source": "mock",
                "competitions_fetched": ["PL"],
                "seasons_fetched": [2023, 2024],
                "total_matches": len(matches),
                "skipped": [],
            }

    monkeypatch.setattr(
        "training.real_dataset.HistoricalFetcher",
        lambda: MockFetcher(),
    )

    X, y, meta = build_real_dataset(force_refresh=True, min_raw_matches=3)
    assert X.shape[0] == len(y)
    assert X.shape[1] == 27
    assert len(y) >= 2
    assert meta["data_source"] == "mock"
    assert set(y.tolist()).issubset({0, 1, 2})

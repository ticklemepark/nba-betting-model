"""Tests for src/features/team/pace.py."""

import pandas as pd
import pytest

from src.features.team.pace import compute_rolling_pace, _compute_per_game_pace


def _make_logs(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "FGM": 40, "FGA": 85, "FG3M": 10, "FG3A": 30,
        "FTM": 15, "FTA": 18, "OREB": 10, "DREB": 30,
        "REB": 40, "AST": 22, "TOV": 12, "STL": 6,
        "BLK": 4, "PF": 18, "PTS": 105, "PLUS_MINUS": 7, "MIN": 48,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestComputePerGamePace:
    def test_regulation_pace(self):
        """poss = 85 - 10 + 12 + 0.44*18 = 94.92; pace = 94.92/48*48 = 94.92"""
        logs = _make_logs([{"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01",
                            "GAME_ID": "G1", "FGA": 85, "OREB": 10, "TOV": 12,
                            "FTA": 18, "MIN": 48}])
        result = _compute_per_game_pace(logs)
        assert result.iloc[0]["PACE"] == pytest.approx(94.92, rel=1e-3)

    def test_pace_column_present(self):
        logs = _make_logs([{"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01",
                            "GAME_ID": "G1"}])
        result = _compute_per_game_pace(logs)
        assert "PACE" in result.columns

    def test_ot_game_higher_possessions_per_48(self):
        """OT game: same possessions but played in 53 min → lower pace per 48."""
        base = {"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01", "GAME_ID": "G1",
                "FGA": 100, "OREB": 10, "TOV": 12, "FTA": 18}
        reg = _make_logs([{**base, "MIN": 48}])
        ot  = _make_logs([{**base, "MIN": 53}])
        reg_pace = _compute_per_game_pace(reg).iloc[0]["PACE"]
        ot_pace  = _compute_per_game_pace(ot).iloc[0]["PACE"]
        assert reg_pace > ot_pace


class TestComputeRollingPace:
    def _setup(self):
        logs = _make_logs([
            {"TEAM": "LAL", "OPP": "GSW", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 85, "OREB": 10, "TOV": 12, "FTA": 18, "MIN": 48},
            {"TEAM": "GSW", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 95, "OREB": 12, "TOV": 14, "FTA": 20, "MIN": 48},
            {"TEAM": "BOS", "OPP": "MIA", "DATE": "2024-01-02", "GAME_ID": "G2",
             "FGA": 80, "OREB": 8, "TOV": 10, "FTA": 15, "MIN": 48},
            {"TEAM": "MIA", "OPP": "BOS", "DATE": "2024-01-02", "GAME_ID": "G2",
             "FGA": 78, "OREB": 9, "TOV": 11, "FTA": 16, "MIN": 48},
        ])
        games = _make_games([
            {"HOME": "LAL", "AWAY": "BOS", "DATE": "2024-01-05", "SEASON": 2024},
        ])
        return logs, games

    def test_adds_home_away_pace_columns(self):
        logs, games = self._setup()
        result = compute_rolling_pace(logs, games, windows=[5])
        assert "HOME_PACE_L5" in result.columns
        assert "AWAY_PACE_L5" in result.columns

    def test_adds_projected_pace_column(self):
        logs, games = self._setup()
        result = compute_rolling_pace(logs, games, windows=[5])
        assert "PROJ_PACE_L5" in result.columns

    def test_projected_pace_is_average(self):
        logs, games = self._setup()
        result = compute_rolling_pace(logs, games, windows=[5])
        row = result.iloc[0]
        expected = (row["HOME_PACE_L5"] + row["AWAY_PACE_L5"]) / 2
        assert row["PROJ_PACE_L5"] == pytest.approx(expected, rel=1e-6)

    def test_multiple_windows(self):
        logs, games = self._setup()
        result = compute_rolling_pace(logs, games, windows=[5, 10])
        for w in [5, 10]:
            assert f"HOME_PACE_L{w}" in result.columns
            assert f"PROJ_PACE_L{w}" in result.columns

    def test_raises_on_missing_log_columns(self):
        _, games = self._setup()
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_pace(pd.DataFrame({"TEAM": ["X"]}), games)

    def test_raises_on_missing_game_columns(self):
        logs, _ = self._setup()
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_pace(logs, pd.DataFrame({"HOME": ["LAL"]}))

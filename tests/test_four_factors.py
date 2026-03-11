"""Tests for src/features/team/four_factors.py."""

import pandas as pd
import pytest

from src.features.team.four_factors import (
    _compute_per_game_factors,
    compute_rolling_four_factors,
)


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


class TestComputePerGameFactors:
    def _two_team_game(self):
        return _make_logs([
            {"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGM": 40, "FGA": 80, "FG3M": 10, "FTA": 20, "TOV": 12,
             "OREB": 10, "DREB": 30},
            {"TEAM": "BOS", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGM": 35, "FGA": 75, "FG3M": 8, "FTA": 18, "TOV": 14,
             "OREB": 8, "DREB": 28},
        ])

    def test_efg_pct_formula(self):
        # LAL: (40 + 0.5*10) / 80 = 45/80 = 0.5625
        logs = self._two_team_game()
        result = _compute_per_game_factors(logs)
        lal = result[result["TEAM"] == "LAL"].iloc[0]
        assert lal["EFG_PCT"] == pytest.approx(0.5625, rel=1e-4)

    def test_tov_pct_is_percentage(self):
        logs = self._two_team_game()
        result = _compute_per_game_factors(logs)
        for _, row in result.iterrows():
            assert 0 <= row["TOV_PCT"] <= 100

    def test_ftr_formula(self):
        # LAL: FTA/FGA = 20/80 = 0.25
        logs = self._two_team_game()
        result = _compute_per_game_factors(logs)
        lal = result[result["TEAM"] == "LAL"].iloc[0]
        assert lal["FTR"] == pytest.approx(0.25, rel=1e-4)

    def test_orb_pct_range(self):
        logs = self._two_team_game()
        result = _compute_per_game_factors(logs)
        for _, row in result.iterrows():
            if not pd.isna(row["ORB_PCT"]):
                assert 0 <= row["ORB_PCT"] <= 100

    def test_orb_pct_uses_opponent_dreb(self):
        """LAL ORB% = LAL_OREB / (LAL_OREB + BOS_DREB) * 100 = 10/(10+28)*100"""
        logs = self._two_team_game()
        result = _compute_per_game_factors(logs)
        lal = result[result["TEAM"] == "LAL"].iloc[0]
        expected = 10 / (10 + 28) * 100
        assert lal["ORB_PCT"] == pytest.approx(expected, rel=1e-4)


class TestComputeRollingFourFactors:
    def _setup(self):
        logs = _make_logs([
            {"TEAM": "LAL", "OPP": "GSW", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGM": 40, "FGA": 80, "FG3M": 10, "FTA": 20, "TOV": 12,
             "OREB": 10, "DREB": 30},
            {"TEAM": "GSW", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGM": 36, "FGA": 82, "FG3M": 12, "FTA": 18, "TOV": 13,
             "OREB": 9, "DREB": 28},
            {"TEAM": "BOS", "OPP": "MIA", "DATE": "2024-01-02", "GAME_ID": "G2",
             "FGM": 38, "FGA": 78, "FG3M": 9, "FTA": 22, "TOV": 11,
             "OREB": 11, "DREB": 32},
            {"TEAM": "MIA", "OPP": "BOS", "DATE": "2024-01-02", "GAME_ID": "G2",
             "FGM": 34, "FGA": 76, "FG3M": 8, "FTA": 16, "TOV": 14,
             "OREB": 8, "DREB": 26},
        ])
        games = _make_games([
            {"HOME": "LAL", "AWAY": "BOS", "DATE": "2024-01-05", "SEASON": 2024},
        ])
        return logs, games

    def test_adds_all_factor_columns(self):
        logs, games = self._setup()
        result = compute_rolling_four_factors(logs, games, windows=[5])
        for side in ["HOME", "AWAY"]:
            for stat in ["EFG_PCT", "TOV_PCT", "ORB_PCT", "FTR"]:
                col = f"{side}_{stat}_L5"
                assert col in result.columns, f"Missing: {col}"

    def test_multiple_windows(self):
        logs, games = self._setup()
        result = compute_rolling_four_factors(logs, games, windows=[5, 10])
        for w in [5, 10]:
            assert f"HOME_EFG_PCT_L{w}" in result.columns

    def test_raises_on_missing_log_columns(self):
        _, games = self._setup()
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_four_factors(pd.DataFrame({"TEAM": ["X"]}), games)

    def test_raises_on_missing_game_columns(self):
        logs, _ = self._setup()
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_four_factors(logs, pd.DataFrame({"HOME": ["LAL"]}))

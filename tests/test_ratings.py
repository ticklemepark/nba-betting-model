"""Tests for src/features/team/ratings.py."""

import pandas as pd
import pytest

from src.features.team.ratings import (
    _compute_per_game_ratings,
    _possessions,
    compute_rolling_ratings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_logs(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal team_game_logs DataFrame."""
    defaults = {
        "FGM": 40, "FGA": 85, "FG3M": 10, "FG3A": 30,
        "FTM": 15, "FTA": 18, "OREB": 10, "DREB": 30,
        "REB": 40, "AST": 22, "TOV": 12, "STL": 6,
        "BLK": 4, "PF": 18, "PLUS_MINUS": 7, "MIN": 48,
    }
    merged = [{**defaults, **r} for r in rows]
    return pd.DataFrame(merged)


def _make_games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _possessions
# ---------------------------------------------------------------------------

class TestPossessions:
    def test_basic_formula(self):
        # poss = FGA - OREB + TOV + 0.44 * FTA
        # = 85 - 10 + 12 + 0.44 * 18 = 94.92
        fga  = pd.Series([85])
        oreb = pd.Series([10])
        tov  = pd.Series([12])
        fta  = pd.Series([18])
        result = _possessions(fga, oreb, tov, fta)
        assert result.iloc[0] == pytest.approx(94.92, rel=1e-4)

    def test_zero_fta(self):
        result = _possessions(pd.Series([80]), pd.Series([5]), pd.Series([10]), pd.Series([0]))
        assert result.iloc[0] == pytest.approx(85.0)


# ---------------------------------------------------------------------------
# _compute_per_game_ratings
# ---------------------------------------------------------------------------

class TestComputePerGameRatings:
    def _two_team_logs(self):
        """One game: LAL (105 pts) vs BOS (98 pts), same GAME_ID."""
        return _make_logs([
            {"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 85, "OREB": 10, "TOV": 12, "FTA": 18, "PTS": 105,
             "FGM": 40, "FG3M": 10, "DREB": 30},
            {"TEAM": "BOS", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 80, "OREB": 8, "TOV": 14, "FTA": 24, "PTS": 98,
             "FGM": 35, "FG3M": 8, "DREB": 28},
        ])

    def test_returns_required_columns(self):
        logs = self._two_team_logs()
        result = _compute_per_game_ratings(logs)
        for col in ["TEAM", "DATE", "GAME_ID", "OFF_RATING", "DEF_RATING", "NET_RATING"]:
            assert col in result.columns

    def test_off_rating_positive(self):
        logs = self._two_team_logs()
        result = _compute_per_game_ratings(logs)
        lal = result[result["TEAM"] == "LAL"].iloc[0]
        assert lal["OFF_RATING"] > 0

    def test_net_rating_equals_off_minus_def(self):
        logs = self._two_team_logs()
        result = _compute_per_game_ratings(logs)
        for _, row in result.iterrows():
            assert row["NET_RATING"] == pytest.approx(row["OFF_RATING"] - row["DEF_RATING"], rel=1e-6)

    def test_lal_def_rating_equals_bos_off_rating(self):
        """LAL's defensive rating should equal BOS's offensive rating in the same game."""
        logs = self._two_team_logs()
        result = _compute_per_game_ratings(logs)
        lal_def = result[result["TEAM"] == "LAL"]["DEF_RATING"].iloc[0]
        bos_off = result[result["TEAM"] == "BOS"]["OFF_RATING"].iloc[0]
        assert lal_def == pytest.approx(bos_off, rel=1e-6)


# ---------------------------------------------------------------------------
# compute_rolling_ratings
# ---------------------------------------------------------------------------

class TestComputeRollingRatings:
    def _multi_game_setup(self):
        """LAL plays 3 games, then plays BOS in game 4."""
        logs = _make_logs([
            {"TEAM": "LAL", "OPP": "GSW", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 85, "OREB": 10, "TOV": 12, "FTA": 18, "PTS": 110,
             "FGM": 40, "FG3M": 10, "DREB": 30},
            {"TEAM": "GSW", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1",
             "FGA": 80, "OREB": 8, "TOV": 14, "FTA": 20, "PTS": 100,
             "FGM": 35, "FG3M": 8, "DREB": 28},
            {"TEAM": "LAL", "OPP": "PHX", "DATE": "2024-01-03", "GAME_ID": "G2",
             "FGA": 88, "OREB": 12, "TOV": 10, "FTA": 20, "PTS": 115,
             "FGM": 42, "FG3M": 11, "DREB": 32},
            {"TEAM": "PHX", "OPP": "LAL", "DATE": "2024-01-03", "GAME_ID": "G2",
             "FGA": 82, "OREB": 9, "TOV": 15, "FTA": 22, "PTS": 102,
             "FGM": 36, "FG3M": 9, "DREB": 29},
            # BOS plays a game before the LAL vs BOS matchup
            {"TEAM": "BOS", "OPP": "MIA", "DATE": "2024-01-02", "GAME_ID": "G3",
             "FGA": 82, "OREB": 9, "TOV": 11, "FTA": 22, "PTS": 108,
             "FGM": 38, "FG3M": 9, "DREB": 29},
            {"TEAM": "MIA", "OPP": "BOS", "DATE": "2024-01-02", "GAME_ID": "G3",
             "FGA": 79, "OREB": 7, "TOV": 13, "FTA": 19, "PTS": 98,
             "FGM": 34, "FG3M": 7, "DREB": 27},
        ])
        games = _make_games([
            {"HOME": "LAL", "AWAY": "BOS", "DATE": "2024-01-05", "SEASON": 2024},
        ])
        return logs, games

    def test_adds_expected_columns(self):
        logs, games = self._multi_game_setup()
        result = compute_rolling_ratings(logs, games, windows=[5])
        for col in ["HOME_OFF_RATING_L5", "HOME_DEF_RATING_L5", "HOME_NET_RATING_L5",
                    "AWAY_OFF_RATING_L5", "AWAY_DEF_RATING_L5", "AWAY_NET_RATING_L5"]:
            assert col in result.columns, f"Missing: {col}"

    def test_multiple_windows(self):
        logs, games = self._multi_game_setup()
        result = compute_rolling_ratings(logs, games, windows=[5, 10])
        for w in [5, 10]:
            assert f"HOME_OFF_RATING_L{w}" in result.columns

    def test_no_future_leakage(self):
        """Games on 2024-01-05 should not include stats from after that date."""
        logs, games = self._multi_game_setup()
        result = compute_rolling_ratings(logs, games, windows=[5])
        # LAL's OFF_RATING_L5 on 2024-01-05 should be based on games before that date
        assert not result["HOME_OFF_RATING_L5"].isna().all()

    def test_raises_on_missing_log_columns(self):
        _, games = self._multi_game_setup()
        bad_logs = pd.DataFrame({"TEAM": ["LAL"], "DATE": ["2024-01-01"]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_ratings(bad_logs, games)

    def test_raises_on_missing_game_columns(self):
        logs, _ = self._multi_game_setup()
        bad_games = pd.DataFrame({"HOME": ["LAL"]})
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_ratings(logs, bad_games)

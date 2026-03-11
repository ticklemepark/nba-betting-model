"""Tests for src/features/pipeline.py."""

import pandas as pd
import pytest

from src.features.pipeline import build_game_features, build_player_features


def _make_games(rows: list[dict]) -> pd.DataFrame:
    defaults = {"HOME": "LAL", "AWAY": "BOS", "DATE": "2024-01-05",
                "SEASON": 2024, "HOME_PTS": 110, "AWAY_PTS": 102}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_team_logs(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01",
        "SEASON": 2024, "IS_HOME": True, "WL": "W", "GAME_ID": "G1",
        "MIN": 48.0, "FGM": 40, "FGA": 85, "FG3M": 10, "FG3A": 30,
        "FTM": 15, "FTA": 18, "OREB": 10, "DREB": 30, "REB": 40,
        "AST": 22, "TOV": 12, "STL": 6, "BLK": 4, "PF": 18,
        "PTS": 110, "PLUS_MINUS": 8,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_player_logs(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "PLAYER_ID": 1, "PLAYER_NAME": "Player A", "TEAM": "LAL",
        "OPP": "BOS", "DATE": "2024-01-01", "SEASON": 2024,
        "IS_HOME": True, "WL": "W", "GAME_ID": "G1",
        "MIN": 32.0, "FGM": 8, "FGA": 16, "FG3M": 2, "FG3A": 6,
        "FTM": 4, "FTA": 5, "OREB": 1, "DREB": 5, "REB": 6,
        "AST": 4, "TOV": 2, "STL": 1, "BLK": 0, "PF": 2,
        "PTS": 22, "PLUS_MINUS": 8,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_player_games(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "PLAYER_ID": 1, "PLAYER_NAME": "Player A", "TEAM": "LAL",
        "OPP": "BOS", "DATE": "2024-01-10", "SEASON": 2024, "IS_HOME": True,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# build_game_features
# ---------------------------------------------------------------------------

class TestBuildGameFeatures:
    def _minimal_setup(self):
        games = _make_games([
            {"HOME": "LAL", "AWAY": "BOS", "DATE": "2024-01-01",
             "HOME_PTS": 110, "AWAY_PTS": 102, "SEASON": 2024},
            {"HOME": "LAL", "AWAY": "GSW", "DATE": "2024-01-05",
             "HOME_PTS": 105, "AWAY_PTS": 98, "SEASON": 2024},
        ])
        team_logs = _make_team_logs([
            {"TEAM": "LAL", "OPP": "BOS", "DATE": "2023-12-28",
             "GAME_ID": "G0", "PTS": 108},
            {"TEAM": "BOS", "OPP": "LAL", "DATE": "2023-12-28",
             "GAME_ID": "G0", "PTS": 100},
            {"TEAM": "GSW", "OPP": "MIA", "DATE": "2023-12-29",
             "GAME_ID": "G00", "PTS": 115},
            {"TEAM": "MIA", "OPP": "GSW", "DATE": "2023-12-29",
             "GAME_ID": "G00", "PTS": 108},
        ])
        return games, team_logs

    def test_returns_dataframe(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert isinstance(result, pd.DataFrame)

    def test_adds_elo_columns(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert "HOME_ELO" in result.columns
        assert "AWAY_ELO" in result.columns

    def test_adds_b2b_columns(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert "HOME_B2B" in result.columns
        assert "AWAY_B2B" in result.columns

    def test_adds_rest_columns(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert "HOME_REST" in result.columns
        assert "AWAY_REST" in result.columns

    def test_adds_rolling_rating_columns(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert "HOME_OFF_RATING_L5" in result.columns
        assert "AWAY_DEF_RATING_L5" in result.columns

    def test_adds_pace_column(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert "PROJ_PACE_L5" in result.columns

    def test_no_rows_dropped(self):
        games, team_logs = self._minimal_setup()
        result = build_game_features(games, team_logs, windows=[5])
        assert len(result) == len(games)

    def test_empty_team_logs_still_returns_phase1_features(self):
        games, _ = self._minimal_setup()
        result = build_game_features(games, pd.DataFrame(), windows=[5])
        assert "HOME_ELO" in result.columns  # Phase 1 features still computed


# ---------------------------------------------------------------------------
# build_player_features
# ---------------------------------------------------------------------------

class TestBuildPlayerFeatures:
    def _setup(self):
        player_logs = _make_player_logs([
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "PTS": 22, "TEAM": "LAL"},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "PTS": 28, "TEAM": "LAL"},
        ])
        team_logs = _make_team_logs([
            {"TEAM": "LAL", "OPP": "BOS", "DATE": "2024-01-01", "GAME_ID": "G1"},
            {"TEAM": "BOS", "OPP": "LAL", "DATE": "2024-01-01", "GAME_ID": "G1"},
        ])
        player_games = _make_player_games([
            {"PLAYER_ID": 1, "DATE": "2024-01-10", "TEAM": "LAL", "OPP": "BOS",
             "SEASON": 2024, "IS_HOME": True},
        ])
        return player_logs, team_logs, player_games

    def test_returns_dataframe(self):
        pl, tl, pg = self._setup()
        result = build_player_features(pg, pl, tl, windows=[5])
        assert isinstance(result, pd.DataFrame)

    def test_adds_rolling_stat_columns(self):
        pl, tl, pg = self._setup()
        result = build_player_features(pg, pl, tl, stats=["PTS"], windows=[5])
        assert "PTS_L5" in result.columns

    def test_adds_usage_columns(self):
        pl, tl, pg = self._setup()
        result = build_player_features(pg, pl, tl, windows=[5])
        assert "USAGE_PROXY_L5" in result.columns

    def test_adds_teammate_boost_column(self):
        pl, tl, pg = self._setup()
        result = build_player_features(pg, pl, tl, absent_players=[], windows=[5])
        assert "TEAMMATE_OUT_BOOST" in result.columns

    def test_empty_player_logs_returns_original(self):
        _, tl, pg = self._setup()
        result = build_player_features(pg, pd.DataFrame(), tl, windows=[5])
        assert len(result) == len(pg)

    def test_no_rows_dropped(self):
        pl, tl, pg = self._setup()
        result = build_player_features(pg, pl, tl, windows=[5])
        assert len(result) == len(pg)

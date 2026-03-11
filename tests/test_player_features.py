"""Tests for src/features/player/* modules."""

import pandas as pd
import pytest

from src.features.player.rolling_stats import compute_rolling_player_stats
from src.features.player.usage import compute_rolling_usage
from src.features.player.home_away import compute_home_away_splits
from src.features.player.matchup import compute_vs_opponent
from src.features.player.availability import compute_teammate_out_boost


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_logs(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "PLAYER_ID": 1, "PLAYER_NAME": "Player A", "TEAM": "LAL",
        "OPP": "BOS", "SEASON": 2024, "IS_HOME": True,
        "WL": "W", "GAME_ID": "G1",
        "MIN": 32.0, "FGM": 8, "FGA": 16, "FG3M": 2, "FG3A": 6,
        "FTM": 4, "FTA": 5, "OREB": 1, "DREB": 5, "REB": 6,
        "AST": 4, "TOV": 2, "STL": 1, "BLK": 0, "PF": 2,
        "PTS": 22, "PLUS_MINUS": 8,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_player_games(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "PLAYER_ID": 1, "PLAYER_NAME": "Player A", "TEAM": "LAL",
        "OPP": "BOS", "DATE": "2024-01-15", "SEASON": 2024, "IS_HOME": True,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# rolling_stats
# ---------------------------------------------------------------------------

class TestComputeRollingPlayerStats:
    def _logs(self):
        return _make_logs([
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "PTS": 20, "REB": 5, "AST": 4},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "PTS": 25, "REB": 7, "AST": 6},
            {"PLAYER_ID": 1, "DATE": "2024-01-05", "PTS": 18, "REB": 4, "AST": 3},
        ])

    def test_adds_rolling_columns(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_player_stats(logs, pg, stats=["PTS"], windows=[3])
        assert "PTS_L3" in result.columns

    def test_adds_season_column(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_player_stats(logs, pg, stats=["PTS"], windows=[3])
        assert "PTS_SEASON" in result.columns

    def test_rolling_mean_correct(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_player_stats(logs, pg, stats=["PTS"], windows=[3])
        # All 3 games before Jan 10 → mean = (20+25+18)/3 = 21.0
        assert result.iloc[0]["PTS_L3"] == pytest.approx(21.0, rel=1e-3)

    def test_no_future_leakage(self):
        logs = self._logs()
        # player_games date is between log games
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-02"}])
        result = compute_rolling_player_stats(logs, pg, stats=["PTS"], windows=[3])
        # Only Jan 1 game is before Jan 2 → mean = 20
        assert result.iloc[0]["PTS_L3"] == pytest.approx(20.0, rel=1e-3)

    def test_raises_on_missing_log_columns(self):
        pg = _make_player_games([{}])
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_player_stats(pd.DataFrame({"X": [1]}), pg, stats=["PTS"])

    def test_raises_on_missing_player_game_columns(self):
        logs = self._logs()
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_player_stats(logs, pd.DataFrame({"X": [1]}), stats=["PTS"])


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------

class TestComputeRollingUsage:
    def _logs(self):
        return _make_logs([
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "FGA": 14, "FTA": 4, "TOV": 2, "MIN": 32},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "FGA": 18, "FTA": 6, "TOV": 3, "MIN": 36},
        ])

    def test_adds_usage_columns(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_usage(logs, pg, windows=[5])
        assert "USAGE_PROXY_L5" in result.columns
        assert "MIN_L5" in result.columns

    def test_adds_season_columns(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_usage(logs, pg, windows=[5])
        assert "USAGE_PROXY_SEASON" in result.columns
        assert "MIN_SEASON" in result.columns

    def test_usage_proxy_positive(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_rolling_usage(logs, pg, windows=[5])
        assert result.iloc[0]["USAGE_PROXY_L5"] > 0

    def test_raises_on_missing_log_columns(self):
        pg = _make_player_games([{}])
        with pytest.raises(ValueError, match="missing columns"):
            compute_rolling_usage(pd.DataFrame({"X": [1]}), pg)


# ---------------------------------------------------------------------------
# home_away
# ---------------------------------------------------------------------------

class TestComputeHomeAwaySplits:
    def _logs(self):
        return _make_logs([
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "IS_HOME": True,  "PTS": 30, "SEASON": 2024},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "IS_HOME": False, "PTS": 18, "SEASON": 2024},
            {"PLAYER_ID": 1, "DATE": "2024-01-05", "IS_HOME": True,  "PTS": 26, "SEASON": 2024},
        ])

    def test_adds_home_away_columns(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_home_away_splits(logs, pg, stats=["PTS"])
        assert "PTS_HOME_AVG" in result.columns
        assert "PTS_AWAY_AVG" in result.columns

    def test_home_avg_higher_for_player_better_at_home(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_home_away_splits(logs, pg, stats=["PTS"])
        assert result.iloc[0]["PTS_HOME_AVG"] > result.iloc[0]["PTS_AWAY_AVG"]

    def test_diff_column_added(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10"}])
        result = compute_home_away_splits(logs, pg, stats=["PTS"])
        assert "PTS_HOME_AWAY_DIFF" in result.columns

    def test_no_future_leakage(self):
        logs = self._logs()
        pg   = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-02"}])
        result = compute_home_away_splits(logs, pg, stats=["PTS"])
        # Only Jan 1 (home) is before Jan 2 → home avg = 30, away = NaN
        assert result.iloc[0]["PTS_HOME_AVG"] == pytest.approx(30.0)

    def test_raises_on_missing_log_columns(self):
        pg = _make_player_games([{}])
        with pytest.raises(ValueError, match="missing columns"):
            compute_home_away_splits(pd.DataFrame({"X": [1]}), pg, stats=["PTS"])


# ---------------------------------------------------------------------------
# matchup
# ---------------------------------------------------------------------------

class TestComputeVsOpponent:
    def _logs(self):
        return _make_logs([
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "OPP": "BOS", "PTS": 30, "SEASON": 2024},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "OPP": "GSW", "PTS": 18, "SEASON": 2024},
            {"PLAYER_ID": 1, "DATE": "2024-01-05", "OPP": "BOS", "PTS": 24, "SEASON": 2024},
        ])

    def test_adds_vs_opp_column(self):
        logs = self._logs()
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10", "OPP": "BOS", "SEASON": 2024}])
        result = compute_vs_opponent(logs, pg, stats=["PTS"])
        assert "PTS_VS_OPP_AVG" in result.columns

    def test_vs_opp_mean_correct(self):
        logs = self._logs()
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10", "OPP": "BOS", "SEASON": 2024}])
        result = compute_vs_opponent(logs, pg, stats=["PTS"])
        # 2 games vs BOS: 30 and 24 → mean = 27
        assert result.iloc[0]["PTS_VS_OPP_AVG"] == pytest.approx(27.0)

    def test_different_opponent_not_mixed(self):
        logs = self._logs()
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10", "OPP": "GSW", "SEASON": 2024}])
        result = compute_vs_opponent(logs, pg, stats=["PTS"])
        # Only 1 game vs GSW: 18
        assert result.iloc[0]["PTS_VS_OPP_AVG"] == pytest.approx(18.0)

    def test_no_prior_games_vs_opp_returns_nan(self):
        logs = self._logs()
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-10", "OPP": "MIA", "SEASON": 2024}])
        result = compute_vs_opponent(logs, pg, stats=["PTS"])
        assert pd.isna(result.iloc[0]["PTS_VS_OPP_AVG"])

    def test_no_future_leakage(self):
        logs = self._logs()
        # Date is Jan 4 → only Jan 1 BOS game counts (Jan 5 BOS game is future)
        pg = _make_player_games([{"PLAYER_ID": 1, "DATE": "2024-01-04", "OPP": "BOS", "SEASON": 2024}])
        result = compute_vs_opponent(logs, pg, stats=["PTS"])
        assert result.iloc[0]["PTS_VS_OPP_AVG"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------

class TestComputeTeammateOutBoost:
    def _logs(self):
        return _make_logs([
            # Player 1 (LAL star, high usage)
            {"PLAYER_ID": 1, "DATE": "2024-01-01", "TEAM": "LAL", "FGA": 20, "FTA": 6, "TOV": 3},
            {"PLAYER_ID": 1, "DATE": "2024-01-03", "TEAM": "LAL", "FGA": 18, "FTA": 5, "TOV": 2},
            # Player 2 (LAL role player)
            {"PLAYER_ID": 2, "DATE": "2024-01-01", "TEAM": "LAL", "FGA": 8, "FTA": 2, "TOV": 1},
            {"PLAYER_ID": 2, "DATE": "2024-01-03", "TEAM": "LAL", "FGA": 10, "FTA": 3, "TOV": 1},
            # Player 3 (absent star on Jan 10)
            {"PLAYER_ID": 3, "DATE": "2024-01-01", "TEAM": "LAL", "FGA": 22, "FTA": 8, "TOV": 4},
            {"PLAYER_ID": 3, "DATE": "2024-01-03", "TEAM": "LAL", "FGA": 20, "FTA": 7, "TOV": 3},
        ])

    def _player_games(self):
        return _make_player_games([
            {"PLAYER_ID": 1, "TEAM": "LAL", "DATE": "2024-01-10"},
            {"PLAYER_ID": 2, "TEAM": "LAL", "DATE": "2024-01-10"},
        ])

    def test_adds_boost_and_flag_columns(self):
        logs  = self._logs()
        pg    = self._player_games()
        result = compute_teammate_out_boost(logs, pg, absent_players=[])
        assert "TEAMMATE_OUT_BOOST" in result.columns
        assert "TEAMMATE_OUT_FLAG" in result.columns

    def test_no_absent_players_returns_zero_boost(self):
        logs = self._logs()
        pg   = self._player_games()
        result = compute_teammate_out_boost(logs, pg, absent_players=[])
        assert (result["TEAMMATE_OUT_BOOST"] == 0.0).all()
        assert (result["TEAMMATE_OUT_FLAG"] == 0).all()

    def test_absent_player_generates_positive_boost(self):
        logs = self._logs()
        pg   = self._player_games()
        absent = [{"player_id": 3, "team": "LAL", "date": "2024-01-10"}]
        result = compute_teammate_out_boost(logs, pg, absent_players=absent)
        # Both active players should get a positive boost
        assert (result["TEAMMATE_OUT_BOOST"] > 0).all()

    def test_flag_set_when_teammate_out(self):
        logs = self._logs()
        pg   = self._player_games()
        absent = [{"player_id": 3, "team": "LAL", "date": "2024-01-10"}]
        result = compute_teammate_out_boost(logs, pg, absent_players=absent)
        assert (result["TEAMMATE_OUT_FLAG"] == 1).all()

    def test_higher_usage_player_gets_more_boost(self):
        """Player 1 has higher usage than Player 2 → gets more of absent star's volume."""
        logs = self._logs()
        pg   = self._player_games()
        absent = [{"player_id": 3, "team": "LAL", "date": "2024-01-10"}]
        result = compute_teammate_out_boost(logs, pg, absent_players=absent)
        p1_boost = result[result["PLAYER_ID"] == 1].iloc[0]["TEAMMATE_OUT_BOOST"]
        p2_boost = result[result["PLAYER_ID"] == 2].iloc[0]["TEAMMATE_OUT_BOOST"]
        assert p1_boost > p2_boost

    def test_raises_on_missing_log_columns(self):
        pg = _make_player_games([{}])
        with pytest.raises(ValueError, match="missing columns"):
            compute_teammate_out_boost(pd.DataFrame({"X": [1]}), pg, absent_players=[])

"""Tests for src/data/nba_api_client.py.

All tests mock the nba_api endpoints — no network calls.
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.nba_api_client import (
    _parse_min,
    _parse_opponent,
    _season_str,
    fetch_player_game_logs,
    fetch_team_game_logs,
)


# ---------------------------------------------------------------------------
# _season_str
# ---------------------------------------------------------------------------

class TestSeasonStr:
    def test_converts_2024(self):
        assert _season_str(2024) == "2023-24"

    def test_converts_2015(self):
        assert _season_str(2015) == "2014-15"

    def test_converts_2020(self):
        assert _season_str(2020) == "2019-20"


# ---------------------------------------------------------------------------
# _parse_opponent
# ---------------------------------------------------------------------------

class TestParseOpponent:
    def test_home_game(self):
        assert _parse_opponent("LAL vs. BOS") == "BOS"

    def test_away_game(self):
        assert _parse_opponent("LAL @ BOS") == "BOS"

    def test_three_letter_abbr(self):
        assert _parse_opponent("GSW vs. PHX") == "PHX"


# ---------------------------------------------------------------------------
# _parse_min
# ---------------------------------------------------------------------------

class TestParseMin:
    def test_numeric_string(self):
        assert _parse_min("48") == pytest.approx(48.0)

    def test_colon_format(self):
        assert _parse_min("48:00") == pytest.approx(48.0)

    def test_ot_colon_format(self):
        assert _parse_min("53:00") == pytest.approx(53.0)

    def test_nan_returns_zero(self):
        assert _parse_min(float("nan")) == 0.0

    def test_none_returns_zero(self):
        assert _parse_min(None) == 0.0


# ---------------------------------------------------------------------------
# fetch_team_game_logs — mocked
# ---------------------------------------------------------------------------

def _make_fake_team_log_df() -> pd.DataFrame:
    """Minimal fake TeamGameLogs response."""
    return pd.DataFrame({
        "TEAM_ABBREVIATION": ["LAL", "BOS"],
        "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
        "GAME_DATE": ["2024-01-01", "2024-01-01"],
        "WL": ["W", "L"],
        "GAME_ID": ["0022400001", "0022400001"],
        "MIN": ["48:00", "48:00"],
        "FGM": [40, 35],
        "FGA": [85, 80],
        "FG3M": [10, 8],
        "FG3A": [30, 25],
        "FTM": [15, 20],
        "FTA": [18, 24],
        "OREB": [10, 8],
        "DREB": [30, 28],
        "REB": [40, 36],
        "AST": [22, 20],
        "TOV": [12, 14],
        "STL": [6, 5],
        "BLK": [4, 3],
        "PF": [18, 20],
        "PTS": [105, 98],
        "PLUS_MINUS": [7, -7],
    })


class TestFetchTeamGameLogs:
    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_returns_clean_dataframe(self, mock_sleep, mock_module):
        mock_endpoint = MagicMock()
        mock_endpoint.get_data_frames.return_value = [_make_fake_team_log_df()]
        mock_module.TeamGameLogs.return_value = mock_endpoint

        result = fetch_team_game_logs(2024)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_team_column_normalized(self, mock_sleep, mock_module):
        mock_endpoint = MagicMock()
        mock_endpoint.get_data_frames.return_value = [_make_fake_team_log_df()]
        mock_module.TeamGameLogs.return_value = mock_endpoint

        result = fetch_team_game_logs(2024)
        assert "TEAM" in result.columns
        assert set(result["TEAM"]) == {"LAL", "BOS"}

    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_opp_column_parsed(self, mock_sleep, mock_module):
        mock_endpoint = MagicMock()
        mock_endpoint.get_data_frames.return_value = [_make_fake_team_log_df()]
        mock_module.TeamGameLogs.return_value = mock_endpoint

        result = fetch_team_game_logs(2024)
        lal_row = result[result["TEAM"] == "LAL"].iloc[0]
        assert lal_row["OPP"] == "BOS"

    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_is_home_column(self, mock_sleep, mock_module):
        mock_endpoint = MagicMock()
        mock_endpoint.get_data_frames.return_value = [_make_fake_team_log_df()]
        mock_module.TeamGameLogs.return_value = mock_endpoint

        result = fetch_team_game_logs(2024)
        lal_row = result[result["TEAM"] == "LAL"].iloc[0]
        bos_row = result[result["TEAM"] == "BOS"].iloc[0]
        assert lal_row["IS_HOME"] is True or lal_row["IS_HOME"] == True
        assert bos_row["IS_HOME"] is False or bos_row["IS_HOME"] == False

    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_api_error_returns_empty(self, mock_sleep, mock_module):
        mock_module.TeamGameLogs.side_effect = Exception("API down")

        result = fetch_team_game_logs(2024)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.data.nba_api_client.teamgamelogs")
    @patch("src.data.nba_api_client.time.sleep")
    def test_season_passed_correctly(self, mock_sleep, mock_module):
        mock_endpoint = MagicMock()
        mock_endpoint.get_data_frames.return_value = [_make_fake_team_log_df()]
        mock_module.TeamGameLogs.return_value = mock_endpoint

        fetch_team_game_logs(2024)
        call_kwargs = mock_module.TeamGameLogs.call_args.kwargs
        assert call_kwargs["season_nullable"] == "2023-24"

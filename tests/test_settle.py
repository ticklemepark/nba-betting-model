"""Tests for scripts/settle_results.py — all nba_api and DB calls are mocked."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import scripts.settle_results as sr
from scripts.settle_results import _get_actual_stat, settle_date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log_df(rows: list[dict]) -> pd.DataFrame:
    """Build a fake nba_api player game log DataFrame."""
    df = pd.DataFrame(rows)
    df["_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    return df


def _make_pending_df(picks: list[dict]) -> pd.DataFrame:
    """Build a fake pending-entries DataFrame matching get_pending_entries output."""
    return pd.DataFrame(picks)


GAME_DATE = date(2026, 3, 8)
SEASON    = "2025-26"


# ---------------------------------------------------------------------------
# _get_actual_stat
# ---------------------------------------------------------------------------

class TestGetActualStat:
    def _fake_df(self, pts=28.0, reb=7.0, ast=5.0):
        return _make_log_df([{
            "GAME_DATE": "MAR 08, 2026",
            "PTS":  pts,
            "REB":  reb,
            "AST":  ast,
            "STL":  2.0,
            "BLK":  1.0,
            "TOV":  3.0,
            "FG3M": 4.0,
            "MIN":  34.0,
        }])

    @patch("scripts.settle_results.nba_players")
    @patch("scripts.settle_results.playergamelog")
    def test_pts_returns_correct_value(self, mock_gl_mod, mock_players):
        mock_players.find_players_by_full_name.return_value = [{"id": 123}]
        mock_inst = MagicMock()
        mock_inst.get_data_frames.return_value = [self._fake_df()]
        mock_gl_mod.PlayerGameLog.return_value = mock_inst

        cache = {}
        result = _get_actual_stat("Jayson Tatum", "PTS", GAME_DATE, SEASON, cache)
        assert result == 28.0

    @patch("scripts.settle_results.nba_players")
    @patch("scripts.settle_results.playergamelog")
    def test_pra_sums_correctly(self, mock_gl_mod, mock_players):
        mock_players.find_players_by_full_name.return_value = [{"id": 123}]
        mock_inst = MagicMock()
        mock_inst.get_data_frames.return_value = [self._fake_df(pts=28, reb=7, ast=5)]
        mock_gl_mod.PlayerGameLog.return_value = mock_inst

        cache = {}
        result = _get_actual_stat("Jayson Tatum", "PRA", GAME_DATE, SEASON, cache)
        assert result == 40.0  # 28 + 7 + 5

    @patch("scripts.settle_results.nba_players")
    @patch("scripts.settle_results.playergamelog")
    def test_caches_player_log(self, mock_gl_mod, mock_players):
        mock_players.find_players_by_full_name.return_value = [{"id": 123}]
        mock_inst = MagicMock()
        mock_inst.get_data_frames.return_value = [self._fake_df()]
        mock_gl_mod.PlayerGameLog.return_value = mock_inst

        cache = {}
        _get_actual_stat("Jayson Tatum", "PTS", GAME_DATE, SEASON, cache)
        _get_actual_stat("Jayson Tatum", "REB", GAME_DATE, SEASON, cache)

        # Second call should NOT hit the API again.
        assert mock_gl_mod.PlayerGameLog.call_count == 1
        assert "Jayson Tatum" in cache

    @patch("scripts.settle_results.nba_players")
    def test_player_not_found_returns_none(self, mock_players):
        mock_players.find_players_by_full_name.return_value = []
        cache = {}
        result = _get_actual_stat("Unknown Player", "PTS", GAME_DATE, SEASON, cache)
        assert result is None

    @patch("scripts.settle_results.nba_players")
    @patch("scripts.settle_results.playergamelog")
    def test_player_did_not_play_returns_none(self, mock_gl_mod, mock_players):
        mock_players.find_players_by_full_name.return_value = [{"id": 123}]
        mock_inst = MagicMock()
        # Game log has only a different date.
        df = _make_log_df([{
            "GAME_DATE": "MAR 07, 2026",
            "PTS": 20.0, "REB": 5.0, "AST": 3.0,
            "STL": 1.0, "BLK": 0.0, "TOV": 2.0, "FG3M": 2.0, "MIN": 30.0,
        }])
        mock_inst.get_data_frames.return_value = [df]
        mock_gl_mod.PlayerGameLog.return_value = mock_inst

        cache = {}
        result = _get_actual_stat("Jayson Tatum", "PTS", GAME_DATE, SEASON, cache)
        assert result is None

    def test_unknown_stat_returns_none(self):
        cache = {}
        result = _get_actual_stat("Jayson Tatum", "BLAH", GAME_DATE, SEASON, cache)
        assert result is None

    @patch("scripts.settle_results.nba_players")
    @patch("scripts.settle_results.playergamelog")
    def test_api_error_returns_none(self, mock_gl_mod, mock_players):
        mock_players.find_players_by_full_name.return_value = [{"id": 123}]
        mock_gl_mod.PlayerGameLog.side_effect = RuntimeError("timeout")

        cache = {}
        result = _get_actual_stat("Jayson Tatum", "PTS", GAME_DATE, SEASON, cache)
        assert result is None
        assert cache["Jayson Tatum"] is None


# ---------------------------------------------------------------------------
# settle_date
# ---------------------------------------------------------------------------

class TestSettleDate:
    """All DB and nba_api calls are mocked."""

    def _pending_df(self, extra_picks: list[dict] | None = None) -> pd.DataFrame:
        picks = [
            {
                "entry_ref":   "aaaa-1111-bbbb-2222",
                "player_name": "Jayson Tatum",
                "stat":        "PTS",
                "direction":   "over",
                "line":        25.5,
                "bet_amount":  50.0,
            }
        ]
        if extra_picks:
            picks.extend(extra_picks)
        return pd.DataFrame(picks)

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_all_picks_hit_entry_won(self, mock_stat, mock_pending, mock_settle):
        mock_pending.return_value = self._pending_df()
        mock_stat.return_value = 30.0   # 30 > 25.5 → hit

        n = settle_date(GAME_DATE, season=SEASON, dry_run=False)
        assert n == 1
        mock_settle.assert_called_once_with("aaaa-1111-bbbb-2222", won=True)

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_pick_missed_entry_lost(self, mock_stat, mock_pending, mock_settle):
        mock_pending.return_value = self._pending_df()
        mock_stat.return_value = 20.0   # 20 < 25.5 → miss

        n = settle_date(GAME_DATE, season=SEASON, dry_run=False)
        assert n == 1
        mock_settle.assert_called_once_with("aaaa-1111-bbbb-2222", won=False)

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_dry_run_does_not_call_settle_entry(self, mock_stat, mock_pending, mock_settle):
        mock_pending.return_value = self._pending_df()
        mock_stat.return_value = 30.0

        n = settle_date(GAME_DATE, season=SEASON, dry_run=True)
        assert n == 1
        mock_settle.assert_not_called()

    @patch("scripts.settle_results.get_pending_entries")
    def test_no_pending_returns_zero(self, mock_pending):
        mock_pending.return_value = pd.DataFrame()

        n = settle_date(GAME_DATE, season=SEASON)
        assert n == 0

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_stat_none_skips_entry(self, mock_stat, mock_pending, mock_settle):
        """If nba_api returns None (player not found), entry is skipped."""
        mock_pending.return_value = self._pending_df()
        mock_stat.return_value = None

        n = settle_date(GAME_DATE, season=SEASON)
        assert n == 0
        mock_settle.assert_not_called()

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_game_winner_pick_skips_entry(self, mock_stat, mock_pending, mock_settle):
        """Entries with stat == 'GAME' (Rival picks) are skipped."""
        df = pd.DataFrame([{
            "entry_ref":   "game-entry-ref",
            "player_name": None,
            "stat":        "GAME",
            "direction":   "home",
            "line":        None,
            "bet_amount":  50.0,
        }])
        mock_pending.return_value = df

        n = settle_date(GAME_DATE, season=SEASON)
        assert n == 0
        mock_settle.assert_not_called()

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_under_pick_hit(self, mock_stat, mock_pending, mock_settle):
        """An UNDER pick hits when actual < line."""
        df = pd.DataFrame([{
            "entry_ref":   "under-entry-ref",
            "player_name": "Player X",
            "stat":        "REB",
            "direction":   "under",
            "line":        8.5,
            "bet_amount":  30.0,
        }])
        mock_pending.return_value = df
        mock_stat.return_value = 6.0   # 6 < 8.5 → hit

        n = settle_date(GAME_DATE, season=SEASON, dry_run=False)
        assert n == 1
        mock_settle.assert_called_once_with("under-entry-ref", won=True)

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_multi_pick_entry_all_hit(self, mock_stat, mock_pending, mock_settle):
        """Multi-pick entry: all picks must hit for entry to be won."""
        df = pd.DataFrame([
            {"entry_ref": "multi-ref", "player_name": "P1", "stat": "PTS",
             "direction": "over", "line": 20.5, "bet_amount": 50.0},
            {"entry_ref": "multi-ref", "player_name": "P2", "stat": "AST",
             "direction": "over", "line": 6.5,  "bet_amount": 50.0},
        ])
        mock_pending.return_value = df
        mock_stat.side_effect = [25.0, 8.0]   # both hit

        n = settle_date(GAME_DATE, season=SEASON, dry_run=False)
        assert n == 1
        mock_settle.assert_called_once_with("multi-ref", won=True)

    @patch("scripts.settle_results.settle_entry")
    @patch("scripts.settle_results.get_pending_entries")
    @patch("scripts.settle_results._get_actual_stat")
    def test_multi_pick_entry_one_miss_lost(self, mock_stat, mock_pending, mock_settle):
        """Multi-pick entry: even one miss means the entire entry is lost."""
        df = pd.DataFrame([
            {"entry_ref": "multi-ref", "player_name": "P1", "stat": "PTS",
             "direction": "over", "line": 20.5, "bet_amount": 50.0},
            {"entry_ref": "multi-ref", "player_name": "P2", "stat": "AST",
             "direction": "over", "line": 6.5,  "bet_amount": 50.0},
        ])
        mock_pending.return_value = df
        mock_stat.side_effect = [25.0, 5.0]   # P1 hits, P2 misses

        n = settle_date(GAME_DATE, season=SEASON, dry_run=False)
        assert n == 1
        mock_settle.assert_called_once_with("multi-ref", won=False)

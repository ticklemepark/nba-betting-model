"""Tests for src/betting/tracker.py — all DB calls are mocked."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.betting.edge_calculator import GamePick, PropPick
from src.betting.tracker import get_pnl_summary, log_entry, settle_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop(player_name="LeBron", team="LAL", opp="BOS", game_id="g1",
          stat="PTS", direction="over", edge=0.10, prob_over=0.60, line=20.0,
          median=25.0, low=18.0, high=32.0):
    return PropPick(
        player_name=player_name, team=team, opp=opp,
        game_id=game_id, stat=stat, direction=direction,
        line=line, model_median=median, model_low=low, model_high=high,
        model_prob_over=prob_over, underdog_prob_over=prob_over - edge,
        edge=edge, game_date=date.today(),
    )


def _game(direction="home", home="LAL", away="BOS", game_id="g1",
          edge=0.12, prob_home=0.62):
    return GamePick(
        game_id=game_id, home_team=home, away_team=away,
        direction=direction, model_prob_home=prob_home,
        underdog_prob_home=prob_home - edge,
        edge=edge, game_date=date.today(),
    )


def _mock_cursor():
    """Return a mock cursor that behaves like psycopg2 DictCursor."""
    cur = MagicMock()
    cur.rowcount = 1
    cur.description = []
    return cur


# ---------------------------------------------------------------------------
# log_entry
# ---------------------------------------------------------------------------

class TestLogEntry:
    @patch("src.betting.tracker.get_cursor")
    def test_returns_uuid_string(self, mock_gc):
        cur = _mock_cursor()
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        ref = log_entry([_prop(), _prop(player_name="AD")],
                        bet_amount=50.0, game_date=date.today())
        assert isinstance(ref, str)
        assert len(ref) == 36  # UUID format

    @patch("src.betting.tracker.get_cursor")
    def test_inserts_entry_and_picks(self, mock_gc):
        cur = _mock_cursor()
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        picks = [_prop(), _prop(player_name="AD"), _game()]
        log_entry(picks, bet_amount=75.0, game_date=date.today())
        # 1 INSERT for bet_entries + 3 INSERTs for entry_picks
        assert cur.execute.call_count == 4

    @patch("src.betting.tracker.get_cursor")
    def test_game_pick_logged_correctly(self, mock_gc):
        cur = _mock_cursor()
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        picks = [_prop(), _game()]
        log_entry(picks, bet_amount=30.0, game_date=date.today())
        # The second execute call for entry_picks should include "GAME" stat
        calls = cur.execute.call_args_list
        # calls[0] = bet_entries insert; calls[1] = prop pick; calls[2] = game pick
        game_call_args = calls[2][0][1]  # positional args of 3rd call
        assert "GAME" in game_call_args

    def test_invalid_entry_size_raises(self):
        with pytest.raises(ValueError, match="Invalid entry size"):
            with patch("src.betting.tracker.get_cursor"):
                log_entry([_prop()], bet_amount=50.0, game_date=date.today())

    def test_too_many_picks_raises(self):
        with pytest.raises(ValueError, match="Invalid entry size"):
            with patch("src.betting.tracker.get_cursor"):
                log_entry([_prop()] * 7, bet_amount=50.0, game_date=date.today())


# ---------------------------------------------------------------------------
# settle_entry
# ---------------------------------------------------------------------------

class TestSettleEntry:
    @patch("src.betting.tracker.get_cursor")
    def test_settle_won(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchone.return_value = {"bet_amount": 100.0, "payout_multiplier": 6.0}
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        settle_entry("fake-ref", won=True)
        update_call = cur.execute.call_args_list[-1]
        params = update_call[0][1]
        assert params[0] == "won"
        assert params[1] == pytest.approx(600.0)  # 100 × 6

    @patch("src.betting.tracker.get_cursor")
    def test_settle_lost(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchone.return_value = {"bet_amount": 50.0, "payout_multiplier": 3.0}
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        settle_entry("fake-ref", won=False)
        update_call = cur.execute.call_args_list[-1]
        params = update_call[0][1]
        assert params[0] == "lost"
        assert params[1] == pytest.approx(-50.0)  # -bet_amount

    @patch("src.betting.tracker.get_cursor")
    def test_missing_entry_raises(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchone.return_value = None
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="not found"):
            settle_entry("nonexistent-ref", won=True)


# ---------------------------------------------------------------------------
# get_pnl_summary
# ---------------------------------------------------------------------------

class TestGetPnlSummary:
    def _mock_rows(self):
        return [
            {"entry_size": 2, "status": "won",  "bet_amount": 100.0, "result_amount": 200.0},
            {"entry_size": 2, "status": "lost", "bet_amount":  50.0, "result_amount": -50.0},
            {"entry_size": 3, "status": "won",  "bet_amount":  75.0, "result_amount": 375.0},
        ]

    @patch("src.betting.tracker.get_cursor")
    def test_computes_correct_roi(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchall.return_value = self._mock_rows()
        cur.description = [
            MagicMock(name="entry_size"), MagicMock(name="status"),
            MagicMock(name="bet_amount"), MagicMock(name="result_amount"),
        ]
        for i, name in enumerate(["entry_size", "status", "bet_amount", "result_amount"]):
            cur.description[i].name = name
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        summary = get_pnl_summary()
        assert summary["n_entries"] == 3
        assert summary["n_won"] == 2
        assert summary["n_lost"] == 1
        assert summary["total_wagered"] == pytest.approx(225.0)
        assert summary["net_pnl"] == pytest.approx(525.0)   # 200 - 50 + 375
        assert summary["roi"] == pytest.approx(525.0 / 225.0, rel=0.01)

    @patch("src.betting.tracker.get_cursor")
    def test_empty_returns_zeros(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchall.return_value = []
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        summary = get_pnl_summary()
        assert summary["n_entries"] == 0
        assert summary["roi"] == 0.0

    @patch("src.betting.tracker.get_cursor")
    def test_by_entry_size_present(self, mock_gc):
        cur = _mock_cursor()
        cur.fetchall.return_value = self._mock_rows()
        desc = []
        for name in ["entry_size", "status", "bet_amount", "result_amount"]:
            col = MagicMock()
            col.name = name
            desc.append(col)
        cur.description = desc
        mock_gc.return_value.__enter__ = lambda s: cur
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        summary = get_pnl_summary()
        assert 2 in summary["by_entry_size"]
        assert 3 in summary["by_entry_size"]

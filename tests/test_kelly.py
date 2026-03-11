"""Tests for src/betting/kelly.py."""

import pytest
import pandas as pd

from src.betting.kelly import (
    UNDERDOG_PAYOUTS,
    _MAX_BET_FRACTION,
    _MIN_BET_FRACTION,
    fractional_kelly,
    size_entry,
    summarise_sizing,
)
from src.betting.edge_calculator import PropPick, GamePick
from datetime import date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop(edge=0.10, prob_over=0.60, direction="over",
          player_name="LeBron", team="LAL", opp="BOS",
          stat="PTS", line=20.0, game_id="g1",
          median=25.0, low=18.0, high=32.0):
    return PropPick(
        player_name=player_name, team=team, opp=opp,
        game_id=game_id, stat=stat, direction=direction,
        line=line, model_median=median, model_low=low, model_high=high,
        model_prob_over=prob_over, underdog_prob_over=prob_over - edge,
        edge=edge, game_date=date.today(),
    )


def _game(edge=0.10, prob_home=0.60, direction="home",
          home="LAL", away="BOS", game_id="g1"):
    return GamePick(
        game_id=game_id, home_team=home, away_team=away,
        direction=direction, model_prob_home=prob_home,
        underdog_prob_home=prob_home - edge,
        edge=edge, game_date=date.today(),
    )


# ---------------------------------------------------------------------------
# fractional_kelly
# ---------------------------------------------------------------------------

class TestFractionalKelly:
    def test_positive_edge_positive_fraction(self):
        f = fractional_kelly(win_prob=0.55, payout=3.0)
        assert f > 0.0

    def test_zero_edge_zero_fraction(self):
        # win_prob = 1/(payout) → Kelly = 0
        payout = 3.0
        f = fractional_kelly(win_prob=1.0 / payout, payout=payout)
        assert f == 0.0

    def test_negative_edge_zero_fraction(self):
        f = fractional_kelly(win_prob=0.30, payout=3.0)
        assert f == 0.0

    def test_capped_at_max_bet_fraction(self):
        # Very strong edge → should be capped
        f = fractional_kelly(win_prob=0.99, payout=3.0, kelly_fraction=1.0)
        assert f <= _MAX_BET_FRACTION

    def test_quarter_kelly_is_smaller(self):
        # win_prob=0.35, payout=3.0 → full Kelly ≈ 0.025 (below _MAX_BET_FRACTION cap)
        full  = fractional_kelly(win_prob=0.35, payout=3.0, kelly_fraction=1.0)
        qkell = fractional_kelly(win_prob=0.35, payout=3.0, kelly_fraction=0.25)
        assert qkell == pytest.approx(full * 0.25, rel=0.01)

    def test_invalid_payout_returns_zero(self):
        assert fractional_kelly(0.60, payout=0.5) == 0.0

    def test_zero_win_prob_returns_zero(self):
        assert fractional_kelly(0.0, payout=3.0) == 0.0

    def test_five_pick_payout(self):
        f = fractional_kelly(win_prob=0.07, payout=UNDERDOG_PAYOUTS[5])
        assert 0.0 <= f <= _MAX_BET_FRACTION


# ---------------------------------------------------------------------------
# size_entry
# ---------------------------------------------------------------------------

class TestSizeEntry:
    def test_two_pick_returns_positive_amount(self):
        picks = [_prop(), _prop(player_name="AD", stat="REB")]
        amount = size_entry(picks, bankroll=1000.0)
        assert amount > 0.0
        assert amount <= 1000.0 * _MAX_BET_FRACTION

    def test_no_edge_returns_zero(self):
        # Win prob = 1/3 for a 2-pick (break-even) → Kelly = 0
        picks = [
            _prop(prob_over=1.0 / 3.0 ** 0.5, edge=0.0),
            _prop(prob_over=1.0 / 3.0 ** 0.5, edge=0.0, player_name="AD"),
        ]
        # With very low win prob, Kelly should be 0
        picks2 = [_prop(prob_over=0.1, edge=0.0), _prop(prob_over=0.1, edge=0.0, player_name="AD")]
        amount = size_entry(picks2, bankroll=1000.0)
        assert amount == 0.0

    def test_invalid_entry_size_returns_zero(self):
        picks = [_prop()] * 7  # 7-pick not supported
        amount = size_entry(picks, bankroll=1000.0)
        assert amount == 0.0

    def test_scales_with_bankroll(self):
        picks = [_prop(), _prop(player_name="AD")]
        a1 = size_entry(picks, bankroll=500.0)
        a2 = size_entry(picks, bankroll=1000.0)
        assert a2 == pytest.approx(a1 * 2.0, rel=0.01)


# ---------------------------------------------------------------------------
# summarise_sizing
# ---------------------------------------------------------------------------

class TestSummariseSizing:
    def test_returns_dataframe(self):
        entries = [
            [_prop(), _prop(player_name="AD")],
            [_prop(), _prop(player_name="AD"), _prop(player_name="CP3", stat="AST")],
        ]
        df = summarise_sizing(entries, bankroll=1000.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_has_expected_columns(self):
        entries = [[_prop(), _prop(player_name="AD")]]
        df = summarise_sizing(entries, bankroll=1000.0)
        for col in ("entry_size", "win_prob", "ev", "payout_multiplier", "bet_amount"):
            assert col in df.columns

    def test_win_prob_in_zero_one(self):
        entries = [[_prop(), _prop(player_name="AD")]]
        df = summarise_sizing(entries, bankroll=1000.0)
        assert 0.0 <= df.iloc[0]["win_prob"] <= 1.0

    def test_empty_entries_returns_empty_df(self):
        df = summarise_sizing([], bankroll=1000.0)
        assert df.empty

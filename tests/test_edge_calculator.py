"""Tests for src/betting/edge_calculator.py."""

from datetime import date

import numpy as np
import pytest

from src.betting.edge_calculator import (
    GamePick,
    PropPick,
    calculate_game_edge,
    calculate_prop_edge,
    prob_over_from_quantiles,
    screen_game_picks,
    screen_prop_picks,
)


# ---------------------------------------------------------------------------
# prob_over_from_quantiles
# ---------------------------------------------------------------------------

class TestProbOverFromQuantiles:
    def test_line_equals_median_returns_half(self):
        # When line == median, P(stat > line) ≈ 0.5
        p = prob_over_from_quantiles(median=20.0, low=15.0, high=25.0, line=20.0)
        assert abs(p - 0.5) < 0.02

    def test_line_well_below_median_returns_high_prob(self):
        p = prob_over_from_quantiles(median=30.0, low=22.0, high=38.0, line=10.0)
        assert p > 0.9

    def test_line_well_above_median_returns_low_prob(self):
        p = prob_over_from_quantiles(median=10.0, low=5.0, high=15.0, line=30.0)
        assert p < 0.1

    def test_degenerate_sigma_step_function_over(self):
        # low == high → sigma = 0
        p = prob_over_from_quantiles(median=20.0, low=20.0, high=20.0, line=19.9)
        assert p == 1.0

    def test_degenerate_sigma_step_function_under(self):
        p = prob_over_from_quantiles(median=20.0, low=20.0, high=20.0, line=20.1)
        assert p == 0.0

    def test_output_in_zero_one(self):
        for line in [0, 10, 20, 30, 50]:
            p = prob_over_from_quantiles(15.0, 10.0, 20.0, line)
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# calculate_prop_edge
# ---------------------------------------------------------------------------

class TestCalculatePropEdge:
    def test_positive_edge_returned(self):
        # Model says 60% OVER, Underdog implies 48% OVER → ~12% edge
        edge = calculate_prop_edge(
            model_median=25.0, model_low=18.0, model_high=32.0,
            underdog_line=20.0, underdog_over_prob=0.48,
            min_edge=0.04,
        )
        assert edge is not None
        assert edge > 0.04

    def test_negative_edge_returned_for_under(self):
        # Model says 30% OVER (→ 70% UNDER), Underdog implies 50%
        edge = calculate_prop_edge(
            model_median=10.0, model_low=5.0, model_high=15.0,
            underdog_line=20.0, underdog_over_prob=0.50,
            min_edge=0.04,
        )
        assert edge is not None
        assert edge < -0.04

    def test_small_edge_returns_none(self):
        # Line at median → ~50% model prob vs 48% underdog → 2% edge < 4% min
        edge = calculate_prop_edge(
            model_median=20.0, model_low=15.0, model_high=25.0,
            underdog_line=20.0, underdog_over_prob=0.48,
            min_edge=0.04,
        )
        assert edge is None

    def test_custom_min_edge(self):
        edge = calculate_prop_edge(
            model_median=20.0, model_low=15.0, model_high=25.0,
            underdog_line=20.0, underdog_over_prob=0.48,
            min_edge=0.00,  # accept all edges
        )
        assert edge is not None


# ---------------------------------------------------------------------------
# calculate_game_edge
# ---------------------------------------------------------------------------

class TestCalculateGameEdge:
    def test_positive_edge_returned(self):
        edge = calculate_game_edge(0.62, 0.50, min_edge=0.04)
        assert edge is not None
        assert abs(edge - 0.12) < 1e-6

    def test_negative_edge_returned(self):
        edge = calculate_game_edge(0.40, 0.52, min_edge=0.04)
        assert edge is not None
        assert edge < 0

    def test_small_edge_returns_none(self):
        edge = calculate_game_edge(0.52, 0.50, min_edge=0.04)
        assert edge is None

    def test_exact_threshold_included(self):
        edge = calculate_game_edge(0.54, 0.50, min_edge=0.04)
        assert edge is not None


# ---------------------------------------------------------------------------
# screen_prop_picks (integration-level, mocked model)
# ---------------------------------------------------------------------------

class _FakePropModel:
    """Minimal PlayerPropModel stub."""
    def predict(self, X):
        n = len(X)
        return {
            "median": np.full(n, 25.0),
            "low":    np.full(n, 18.0),
            "high":   np.full(n, 32.0),
        }


def _make_prop_line(player_name="LeBron James", team="LAL", opp="BOS",
                    game_id="g1", stat="PTS", line=20.0,
                    over_payout=0.48, game_date=None):
    from src.data.scrapers.underdog import UnderdogPropLine
    return UnderdogPropLine(
        player_id=player_name.replace(" ", "_").lower(),
        player_name=player_name,
        team=team,
        opp=opp,
        game_id=game_id,
        stat=stat,
        line=line,
        over_payout=over_payout,
        under_payout=1.0 - over_payout,
        game_date=game_date or date.today(),
    )


class TestScreenPropPicks:
    def _player_df(self, player_name="LeBron James", team="LAL", opp="BOS",
                   game_id="g1", stat="PTS"):
        import pandas as pd
        return pd.DataFrame([{
            "PLAYER_NAME": player_name,
            "TEAM":        team,
            "OPP":         opp,
            "GAME_ID":     game_id,
            f"{stat}_L5":  22.0,
            f"{stat}_L10": 23.0,
            f"{stat}_L20": 21.5,
            f"{stat}_SEASON": 22.0,
            "MIN_L5":       34.0,
            "MIN_L10":      33.5,
            "MIN_L20":      33.0,
            "IS_HOME":      1,
        }])

    def test_returns_picks_with_edge(self):
        df = self._player_df()
        line = _make_prop_line(line=20.0, over_payout=0.48)
        picks = screen_prop_picks(df, {"PTS": _FakePropModel()}, [line], min_edge=0.04)
        assert len(picks) == 1
        assert picks[0].direction == "over"
        assert picks[0].edge > 0.04

    def test_empty_df_returns_empty(self):
        import pandas as pd
        picks = screen_prop_picks(pd.DataFrame(), {"PTS": _FakePropModel()}, [], min_edge=0.04)
        assert picks == []

    def test_no_model_for_stat_skips(self):
        df = self._player_df(stat="PTS")
        line = _make_prop_line(stat="REB", line=8.0, over_payout=0.48)
        picks = screen_prop_picks(df, {"PTS": _FakePropModel()}, [line], min_edge=0.04)
        assert picks == []

    def test_unknown_player_skips(self):
        df = self._player_df(player_name="LeBron James")
        line = _make_prop_line(player_name="Unknown Player")
        picks = screen_prop_picks(df, {"PTS": _FakePropModel()}, [line], min_edge=0.04)
        assert picks == []

    def test_sorted_by_edge_descending(self):
        import pandas as pd
        from src.data.scrapers.underdog import UnderdogPropLine

        df = pd.DataFrame([
            {"PLAYER_NAME": "Player A", "TEAM": "LAL", "OPP": "BOS",
             "GAME_ID": "g1", "PTS_L5": 22.0, "PTS_L10": 23.0,
             "PTS_L20": 21.5, "PTS_SEASON": 22.0,
             "MIN_L5": 34.0, "MIN_L10": 33.5, "MIN_L20": 33.0, "IS_HOME": 1},
            {"PLAYER_NAME": "Player B", "TEAM": "LAL", "OPP": "BOS",
             "GAME_ID": "g1", "PTS_L5": 22.0, "PTS_L10": 23.0,
             "PTS_L20": 21.5, "PTS_SEASON": 22.0,
             "MIN_L5": 34.0, "MIN_L10": 33.5, "MIN_L20": 33.0, "IS_HOME": 1},
        ])
        lines = [
            UnderdogPropLine("pa", "Player A", "LAL", "BOS", "g1", "PTS", 18.0, 0.48, 0.52, date.today()),
            UnderdogPropLine("pb", "Player B", "LAL", "BOS", "g1", "PTS", 22.0, 0.48, 0.52, date.today()),
        ]
        picks = screen_prop_picks(df, {"PTS": _FakePropModel()}, lines, min_edge=0.00)
        if len(picks) >= 2:
            assert abs(picks[0].edge) >= abs(picks[1].edge)


# ---------------------------------------------------------------------------
# screen_game_picks (integration-level)
# ---------------------------------------------------------------------------

class _FakeGameModel:
    def predict_proba(self, X):
        return np.array([[0.38, 0.62]] * len(X))


def _make_game_line(home="LAL", away="BOS", home_payout=0.50, game_id="g1",
                    game_date=None):
    from src.data.scrapers.underdog import UnderdogGameLine
    return UnderdogGameLine(
        game_id=game_id, home_team=home, away_team=away,
        home_payout=home_payout, away_payout=1.0 - home_payout,
        game_date=game_date or date.today(),
    )


class TestScreenGamePicks:
    def _game_df(self, home="LAL", away="BOS"):
        import pandas as pd
        return pd.DataFrame([{
            "HOME": home, "AWAY": away,
            "HOME_ELO": 1550.0, "AWAY_ELO": 1480.0, "ELO_DIFF": 70.0,
        }])

    def test_returns_game_pick(self):
        df = self._game_df()
        line = _make_game_line(home_payout=0.50)
        picks = screen_game_picks(df, _FakeGameModel(), [line], min_edge=0.04)
        assert len(picks) == 1
        assert picks[0].direction == "home"
        assert picks[0].edge > 0.04

    def test_empty_df_returns_empty(self):
        import pandas as pd
        picks = screen_game_picks(pd.DataFrame(), _FakeGameModel(), [], min_edge=0.04)
        assert picks == []

    def test_unmatched_game_skipped(self):
        df = self._game_df(home="LAL", away="BOS")
        line = _make_game_line(home="GSW", away="NYK")
        picks = screen_game_picks(df, _FakeGameModel(), [line], min_edge=0.04)
        assert picks == []

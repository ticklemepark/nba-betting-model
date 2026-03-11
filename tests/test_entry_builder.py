"""Tests for src/betting/entry_builder.py."""

from datetime import date

import pytest

from src.betting.edge_calculator import GamePick, PropPick
from src.betting.entry_builder import (
    _PACE_OVER_BONUS,
    _SAME_GAME_OPPOSITE_PEN,
    _SAME_TEAM_OVER_BONUS,
    build_entries,
    rank_entries,
    score_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop(player_name="LeBron", team="LAL", opp="BOS", game_id="g1",
          stat="PTS", direction="over", edge=0.10, prob_over=0.60,
          line=20.0, median=25.0, low=18.0, high=32.0):
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


# ---------------------------------------------------------------------------
# score_correlation
# ---------------------------------------------------------------------------

class TestScoreCorrelation:
    def test_same_team_both_over_counting_stats(self):
        a = _prop(team="LAL", direction="over", stat="PTS")
        b = _prop(player_name="AD", team="LAL", direction="over", stat="REB")
        assert score_correlation(a, b) == _SAME_TEAM_OVER_BONUS

    def test_same_team_both_under_counting_stats(self):
        a = _prop(team="LAL", direction="under", stat="PTS")
        b = _prop(player_name="AD", team="LAL", direction="under", stat="REB")
        assert score_correlation(a, b) == _SAME_TEAM_OVER_BONUS

    def test_different_teams_both_over_pace_bonus(self):
        a = _prop(team="LAL", direction="over", stat="PTS")
        b = _prop(player_name="Tatum", team="BOS", opp="LAL",
                  direction="over", stat="AST", game_id="g1")
        assert score_correlation(a, b) == _PACE_OVER_BONUS

    def test_same_game_opposite_direction_penalty(self):
        a = _prop(team="LAL", direction="over",  stat="PTS")
        b = _prop(player_name="AD", team="LAL", direction="under", stat="REB")
        assert score_correlation(a, b) == _SAME_GAME_OPPOSITE_PEN

    def test_different_games_zero_correlation(self):
        a = _prop(game_id="g1")
        b = _prop(player_name="Tatum", game_id="g2")
        assert score_correlation(a, b) == 0.0

    def test_two_game_picks_zero_correlation(self):
        g1 = _game(game_id="g1")
        g2 = _game(game_id="g2", home="GSW", away="NYK")
        assert score_correlation(g1, g2) == 0.0

    def test_game_pick_and_same_game_prop_blowout_stack(self):
        g = _game(direction="home", home="LAL", away="BOS", game_id="g1")
        # Favourite's player going UNDER (blowout stack)
        p = _prop(team="LAL", direction="under", stat="PTS", game_id="g1")
        corr = score_correlation(g, p)
        assert corr > 0

    def test_game_pick_and_same_game_prop_inverse(self):
        g = _game(direction="home", home="LAL", away="BOS", game_id="g1")
        # Underdog's player going OVER (garbage time volume)
        p = _prop(player_name="Tatum", team="BOS", direction="over",
                  stat="PTS", game_id="g1")
        corr = score_correlation(g, p)
        assert corr > 0

    def test_symmetry(self):
        a = _prop(team="LAL", direction="over", stat="PTS")
        b = _prop(player_name="AD", team="LAL", direction="over", stat="REB")
        assert score_correlation(a, b) == score_correlation(b, a)


# ---------------------------------------------------------------------------
# build_entries
# ---------------------------------------------------------------------------

class TestBuildEntries:
    def _pool(self, n=5):
        picks = []
        for i in range(n):
            picks.append(_prop(
                player_name=f"Player{i}",
                game_id=f"g{i % 2}",
                edge=0.05 + 0.01 * i,
            ))
        return picks

    def test_returns_list_of_lists(self):
        pool = self._pool(4)
        entries = build_entries(pool, min_picks=2, max_picks=3)
        assert isinstance(entries, list)
        assert all(isinstance(e, list) for e in entries)

    def test_entry_sizes_in_range(self):
        pool = self._pool(5)
        entries = build_entries(pool, min_picks=2, max_picks=4)
        for entry in entries:
            assert 2 <= len(entry) <= 4

    def test_max_entries_respected(self):
        pool = self._pool(10)
        entries = build_entries(pool, max_entries=5)
        assert len(entries) <= 5

    def test_insufficient_picks_returns_empty(self):
        entries = build_entries([_prop()], min_picks=2)
        assert entries == []

    def test_no_duplicate_entries(self):
        pool = self._pool(6)
        entries = build_entries(pool, max_entries=50)
        keys = [frozenset(id(p) for p in e) for e in entries]
        assert len(keys) == len(set(keys))

    def test_picks_are_subset_of_pool(self):
        pool = self._pool(5)
        entries = build_entries(pool)
        pool_ids = {id(p) for p in pool}
        for entry in entries:
            for pick in entry:
                assert id(pick) in pool_ids


# ---------------------------------------------------------------------------
# rank_entries
# ---------------------------------------------------------------------------

class TestRankEntries:
    def test_returns_sorted_descending(self):
        pool = [
            _prop(edge=0.05),
            _prop(player_name="AD", edge=0.15, stat="REB"),
        ]
        entries = build_entries(pool, min_picks=2, max_picks=2)
        ranked = rank_entries(entries)
        if len(ranked) >= 2:
            scores = [s for _, s in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_returns_tuple_pairs(self):
        pool = [_prop(), _prop(player_name="AD")]
        entries = build_entries(pool, min_picks=2, max_picks=2)
        ranked = rank_entries(entries)
        for item in ranked:
            assert len(item) == 2
            picks_list, score = item
            assert isinstance(picks_list, list)
            assert isinstance(score, float)

    def test_empty_entries_returns_empty(self):
        ranked = rank_entries([])
        assert ranked == []

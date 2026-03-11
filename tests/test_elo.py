"""Tests for src/features/team/elo.py.

Structure: one happy-path test and one most-likely-failure-mode test per
function, as required by the engineering rules.
"""

import pandas as pd
import pytest

from src.features.team.elo import (
    HOME_COURT_ADVANTAGE,
    SEASON_REGRESSION_TARGET,
    SEASON_REGRESSION_WEIGHT,
    STARTING_ELO,
    apply_season_regression,
    compute_elo_ratings,
    elo_k,
    update_elo,
    win_probs,
)


# ---------------------------------------------------------------------------
# win_probs
# ---------------------------------------------------------------------------

class TestWinProbs:
    def test_probs_sum_to_one(self):
        home_p, away_p = win_probs(1500, 1500)
        assert abs(home_p + away_p - 1.0) < 1e-10

    def test_home_court_advantage_favors_home(self):
        # Equal ELO teams: home should win > 50% due to HCA
        home_p, away_p = win_probs(1500, 1500, HOME_COURT_ADVANTAGE)
        assert home_p > 0.50

    def test_stronger_team_has_higher_prob(self):
        home_p, _ = win_probs(home_elo=1600, away_elo=1400)
        _, away_p = win_probs(home_elo=1400, away_elo=1600)
        # Both benefit from HCA when home, but the 200-pt ELO diff is large
        assert home_p > away_p

    def test_zero_home_court_advantage_gives_50_50_equal_elos(self):
        home_p, away_p = win_probs(1500, 1500, home_court_advantage=0)
        assert abs(home_p - 0.50) < 1e-10
        assert abs(away_p - 0.50) < 1e-10


# ---------------------------------------------------------------------------
# elo_k
# ---------------------------------------------------------------------------

class TestEloK:
    def test_blowout_produces_larger_k_than_close_game(self):
        k_blowout = elo_k(mov=30, elo_diff=0)
        k_close = elo_k(mov=3, elo_diff=0)
        assert k_blowout > k_close

    def test_k_is_positive_for_home_loss(self):
        # Away team wins (negative MOV from home perspective) — K must still be positive
        k = elo_k(mov=-10, elo_diff=50)
        assert k > 0

    def test_k_is_symmetric_around_zero_elo_diff(self):
        # With elo_diff=0, a home +15 win and away +15 win should give same K
        k_home_win = elo_k(mov=15, elo_diff=0)
        k_away_win = elo_k(mov=-15, elo_diff=0)
        assert abs(k_home_win - k_away_win) < 1e-10

    def test_expected_k_magnitude(self):
        # For a close game (MOV=5) with equal teams, K should be near BASE_K (20)
        # (5+3)^0.8 / 7.5 ≈ 5.78 / 7.5 ≈ 0.77; 20 * 0.77 ≈ 15.4
        k = elo_k(mov=5, elo_diff=0)
        assert 10 < k < 30


# ---------------------------------------------------------------------------
# update_elo
# ---------------------------------------------------------------------------

class TestUpdateElo:
    def test_winner_elo_increases(self):
        new_home, new_away = update_elo(
            home_score=110, away_score=100,
            home_elo=1500, away_elo=1500,
        )
        assert new_home > 1500
        assert new_away < 1500

    def test_away_winner_elo_increases(self):
        new_home, new_away = update_elo(
            home_score=95, away_score=108,
            home_elo=1500, away_elo=1500,
        )
        assert new_away > 1500
        assert new_home < 1500

    def test_elo_sum_is_conserved(self):
        # ELO is a zero-sum system: points gained == points lost
        h0, a0 = 1540.0, 1460.0
        new_h, new_a = update_elo(110, 100, h0, a0)
        assert abs((new_h + new_a) - (h0 + a0)) < 1e-8

    def test_upset_moves_elo_more_than_expected_result(self):
        # Weak home team (1400) beating strong away team (1600) = big upset
        new_h_upset, _ = update_elo(105, 100, home_elo=1400, away_elo=1600)
        # Strong home team (1600) beating weak away team (1400) = expected result
        new_h_expected, _ = update_elo(105, 100, home_elo=1600, away_elo=1400)
        # Upset should earn more ELO points
        assert (new_h_upset - 1400) > (new_h_expected - 1600)


# ---------------------------------------------------------------------------
# apply_season_regression
# ---------------------------------------------------------------------------

class TestApplySeasonRegression:
    def test_above_average_team_regresses_down(self):
        regressed = apply_season_regression(1600)
        assert regressed < 1600
        assert regressed > SEASON_REGRESSION_TARGET

    def test_below_average_team_regresses_up(self):
        regressed = apply_season_regression(1400)
        assert regressed > 1400
        assert regressed < SEASON_REGRESSION_TARGET

    def test_regression_formula(self):
        elo = 1620.0
        expected = (1 - SEASON_REGRESSION_WEIGHT) * elo + SEASON_REGRESSION_WEIGHT * SEASON_REGRESSION_TARGET
        assert abs(apply_season_regression(elo) - expected) < 1e-10


# ---------------------------------------------------------------------------
# compute_elo_ratings
# ---------------------------------------------------------------------------

def _make_games(*games: tuple) -> pd.DataFrame:
    """Helper: build a minimal games DataFrame from (home, away, h_pts, a_pts, date, season) tuples."""
    rows = [
        {"HOME": h, "AWAY": a, "HOME_PTS": hp, "AWAY_PTS": ap, "DATE": d, "SEASON": s}
        for h, a, hp, ap, d, s in games
    ]
    return pd.DataFrame(rows)


class TestComputeEloRatings:
    def test_output_has_required_columns(self):
        games = _make_games(
            ("LAL", "BOS", 110, 105, "2024-01-01", 2024),
        )
        result = compute_elo_ratings(games)
        for col in ("HOME_ELO", "AWAY_ELO", "HOME_AFTER", "AWAY_AFTER"):
            assert col in result.columns

    def test_new_teams_start_at_1500(self):
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
        )
        result = compute_elo_ratings(games)
        assert result.iloc[0]["HOME_ELO"] == STARTING_ELO
        assert result.iloc[0]["AWAY_ELO"] == STARTING_ELO

    def test_no_data_leakage_pre_game_elo(self):
        # Game 1: LAL beats BOS → LAL ELO rises.  Game 2: LAL ELO entering
        # should equal the POST value from game 1 — NOT the post value of game 2.
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "GSW", 115, 108, "2024-01-03", 2024),
        )
        result = compute_elo_ratings(games)
        assert result.iloc[1]["HOME_ELO"] == result.iloc[0]["HOME_AFTER"]

    def test_season_regression_applied_in_new_season(self):
        # LAL plays in 2023, then plays again in 2024.
        # Their 2024 entry ELO must be the regressed value, not the raw 2023 ending ELO.
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2023-04-01", 2023),
            ("LAL", "GSW", 108, 105, "2024-01-15", 2024),
        )
        result = compute_elo_ratings(games)
        elo_end_2023 = result.iloc[0]["HOME_AFTER"]
        elo_start_2024 = result.iloc[1]["HOME_ELO"]
        expected = apply_season_regression(elo_end_2023)
        assert abs(elo_start_2024 - expected) < 1e-8

    def test_missing_columns_raises_value_error(self):
        bad_df = pd.DataFrame({"HOME": ["LAL"], "AWAY": ["BOS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_elo_ratings(bad_df)

    def test_games_sorted_by_date_regardless_of_input_order(self):
        # Input is reversed — earlier game is second row.
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-10", 2024),
            ("LAL", "GSW", 108, 105, "2024-01-01", 2024),  # earlier game, input second
        )
        result = compute_elo_ratings(games)
        # After sorting, GSW game comes first, so LAL's ELO in the BOS game
        # must already reflect the GSW result.
        gsw_row = result[result["AWAY"] == "GSW"].iloc[0]
        bos_row = result[result["AWAY"] == "BOS"].iloc[0]
        assert bos_row["HOME_ELO"] == gsw_row["HOME_AFTER"]

    def test_elo_sum_conserved_across_all_games(self):
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("GSW", "LAL", 105, 112, "2024-01-03", 2024),
            ("BOS", "GSW", 98, 101, "2024-01-05", 2024),
        )
        result = compute_elo_ratings(games)
        for _, row in result.iterrows():
            pre_sum = row["HOME_ELO"] + row["AWAY_ELO"]
            post_sum = row["HOME_AFTER"] + row["AWAY_AFTER"]
            assert abs(pre_sum - post_sum) < 1e-6

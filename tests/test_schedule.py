"""Tests for src/features/team/schedule.py."""

import pandas as pd
import pytest

from src.features.team.schedule import compute_b2b_flags, compute_rest_days


def _make_games(*games: tuple) -> pd.DataFrame:
    rows = [
        {"HOME": h, "AWAY": a, "DATE": d}
        for h, a, d in games
    ]
    return pd.DataFrame(rows)


class TestComputeB2BFlags:
    def test_output_has_required_columns(self):
        games = _make_games(("LAL", "BOS", "2024-01-01"))
        result = compute_b2b_flags(games)
        assert "HOME_B2B" in result.columns
        assert "AWAY_B2B" in result.columns

    def test_first_game_is_never_b2b(self):
        games = _make_games(("LAL", "BOS", "2024-01-01"))
        result = compute_b2b_flags(games)
        assert result.iloc[0]["HOME_B2B"] == 0
        assert result.iloc[0]["AWAY_B2B"] == 0

    def test_consecutive_days_is_b2b(self):
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-02"),  # LAL plays next day
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["HOME_B2B"] == 1

    def test_two_day_gap_is_not_b2b(self):
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-03"),  # one day rest
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["HOME_B2B"] == 0

    def test_month_boundary_handled_correctly(self):
        # Jan 31 → Feb 1 is consecutive (1 day apart)
        games = _make_games(
            ("LAL", "BOS", "2024-01-31"),
            ("LAL", "GSW", "2024-02-01"),
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["HOME_B2B"] == 1

    def test_year_boundary_handled_correctly(self):
        # Dec 31 → Jan 1 is consecutive
        games = _make_games(
            ("LAL", "BOS", "2023-12-31"),
            ("LAL", "GSW", "2024-01-01"),
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["HOME_B2B"] == 1

    def test_away_team_b2b_tracked_independently(self):
        # BOS plays Jan 1, then travels and plays as away on Jan 2
        games = _make_games(
            ("BOS", "MIA", "2024-01-01"),   # BOS at home
            ("LAL", "BOS", "2024-01-02"),   # BOS on the road next day
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["AWAY_B2B"] == 1
        assert result.iloc[1]["HOME_B2B"] == 0  # LAL has not played recently

    def test_only_the_team_that_played_yesterday_is_flagged(self):
        # LAL plays Jan 1; BOS has not played at all
        games = _make_games(
            ("LAL", "GSW", "2024-01-01"),
            ("LAL", "BOS", "2024-01-02"),
        )
        result = compute_b2b_flags(games)
        assert result.iloc[1]["HOME_B2B"] == 1   # LAL on B2B
        assert result.iloc[1]["AWAY_B2B"] == 0   # BOS first game

    def test_missing_columns_raises_value_error(self):
        bad_df = pd.DataFrame({"HOME": ["LAL"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_b2b_flags(bad_df)

    def test_games_sorted_by_date_regardless_of_input_order(self):
        # Input reversed
        games = _make_games(
            ("LAL", "BOS", "2024-01-02"),
            ("LAL", "GSW", "2024-01-01"),  # earlier, input second
        )
        result = compute_b2b_flags(games)
        gsw_row = result[result["AWAY"] == "GSW"].iloc[0]
        bos_row = result[result["AWAY"] == "BOS"].iloc[0]
        assert gsw_row["HOME_B2B"] == 0   # first game for LAL
        assert bos_row["HOME_B2B"] == 1   # LAL played yesterday


class TestComputeRestDays:
    def test_output_has_required_columns(self):
        games = _make_games(("LAL", "BOS", "2024-01-01"))
        result = compute_rest_days(games)
        assert "HOME_REST" in result.columns
        assert "AWAY_REST" in result.columns

    def test_first_game_returns_cap(self):
        games = _make_games(("LAL", "BOS", "2024-01-01"))
        result = compute_rest_days(games)
        assert result.iloc[0]["HOME_REST"] == 7
        assert result.iloc[0]["AWAY_REST"] == 7

    def test_b2b_is_one_day_rest(self):
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-02"),
        )
        result = compute_rest_days(games)
        assert result.iloc[1]["HOME_REST"] == 1

    def test_two_day_gap_is_two_days_rest(self):
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-03"),
        )
        result = compute_rest_days(games)
        assert result.iloc[1]["HOME_REST"] == 2

    def test_long_gap_capped_at_seven(self):
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-15"),  # 14 days later
        )
        result = compute_rest_days(games)
        assert result.iloc[1]["HOME_REST"] == 7

    def test_home_and_away_tracked_independently(self):
        games = _make_games(
            ("BOS", "MIA", "2024-01-01"),
            ("LAL", "BOS", "2024-01-03"),  # BOS away, 2 days rest; LAL first game
        )
        result = compute_rest_days(games)
        assert result.iloc[1]["AWAY_REST"] == 2   # BOS had 2 days rest
        assert result.iloc[1]["HOME_REST"] == 7   # LAL first game

    def test_consistent_with_b2b_flag(self):
        """rest_days == 1 must align with B2B == 1."""
        games = _make_games(
            ("LAL", "BOS", "2024-01-01"),
            ("LAL", "GSW", "2024-01-02"),
        )
        b2b  = compute_b2b_flags(games)
        rest = compute_rest_days(games)
        # Sort both by date and compare
        b2b_sorted  = b2b.sort_values("DATE").reset_index(drop=True)
        rest_sorted = rest.sort_values("DATE").reset_index(drop=True)
        assert (b2b_sorted["HOME_B2B"] == 1).equals(rest_sorted["HOME_REST"] == 1)

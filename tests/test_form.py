"""Tests for src/features/team/form.py."""

import pandas as pd
import pytest

from src.features.team.form import compute_streaks, compute_wins_rolling


def _make_games(*games: tuple) -> pd.DataFrame:
    rows = [
        {"HOME": h, "AWAY": a, "HOME_PTS": hp, "AWAY_PTS": ap, "DATE": d, "SEASON": s}
        for h, a, hp, ap, d, s in games
    ]
    return pd.DataFrame(rows)


class TestComputeStreaks:
    def test_new_teams_start_at_zero(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_streaks(games)
        assert result.iloc[0]["HOME_STREAK"] == 0
        assert result.iloc[0]["AWAY_STREAK"] == 0

    def test_output_has_required_columns(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_streaks(games)
        assert "HOME_STREAK" in result.columns
        assert "AWAY_STREAK" in result.columns

    def test_consecutive_wins_accumulate(self):
        # LAL wins game 1 and game 2 → enters game 3 with streak of 2
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "GSW", 108, 102, "2024-01-03", 2024),
            ("LAL", "MIA", 105, 98,  "2024-01-05", 2024),
        )
        result = compute_streaks(games)
        assert result.iloc[0]["HOME_STREAK"] == 0  # no prior games
        assert result.iloc[1]["HOME_STREAK"] == 1  # won game 1
        assert result.iloc[2]["HOME_STREAK"] == 2  # won games 1+2

    def test_loss_resets_streak_to_zero(self):
        # LAL wins then loses → streak should be 0 before the 3rd game
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),  # LAL wins
            ("LAL", "GSW",  90, 105, "2024-01-03", 2024),  # LAL loses
            ("LAL", "MIA", 100,  95, "2024-01-05", 2024),  # check streak
        )
        result = compute_streaks(games)
        assert result.iloc[2]["HOME_STREAK"] == 0

    def test_season_boundary_resets_streak(self):
        # LAL wins in 2023, then appears in 2024 → streak resets
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2023-04-01", 2023),
            ("LAL", "GSW", 108, 102, "2023-04-03", 2023),
            ("LAL", "MIA", 105,  98, "2024-01-05", 2024),  # new season
        )
        result = compute_streaks(games)
        assert result.iloc[2]["HOME_STREAK"] == 0

    def test_away_team_streak_tracked_independently(self):
        # BOS wins as away team → their streak carries to next game
        games = _make_games(
            ("LAL", "BOS",  95, 110, "2024-01-01", 2024),  # BOS wins away
            ("GSW", "BOS", 100, 100, "2024-01-03", 2024),  # BOS enters with streak 1
        )
        result = compute_streaks(games)
        assert result.iloc[1]["AWAY_STREAK"] == 1

    def test_no_data_leakage(self):
        # Streak entering game N must equal the post-game streak from game N-1
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "GSW", 108, 102, "2024-01-03", 2024),
        )
        result = compute_streaks(games)
        # After game 1 LAL has streak 1; they should enter game 2 with streak 1
        assert result.iloc[1]["HOME_STREAK"] == 1

    def test_missing_columns_raises_value_error(self):
        bad_df = pd.DataFrame({"HOME": ["LAL"], "AWAY": ["BOS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_streaks(bad_df)

    def test_games_sorted_by_date_regardless_of_input_order(self):
        # Input reversed — earlier game second
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-05", 2024),
            ("LAL", "GSW", 108, 102, "2024-01-01", 2024),  # earlier, input second
        )
        result = compute_streaks(games)
        gsw_row = result[result["AWAY"] == "GSW"].iloc[0]
        bos_row = result[result["AWAY"] == "BOS"].iloc[0]
        # GSW game processed first → LAL enters BOS game with streak 1
        assert gsw_row["HOME_STREAK"] == 0
        assert bos_row["HOME_STREAK"] == 1


class TestComputeWinsRolling:
    def test_output_has_required_columns(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_wins_rolling(games, window=10)
        assert "HOME_WINS_L10" in result.columns
        assert "AWAY_WINS_L10" in result.columns

    def test_first_game_is_zero_wins(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_wins_rolling(games, window=10)
        assert result.iloc[0]["HOME_WINS_L10"] == 0
        assert result.iloc[0]["AWAY_WINS_L10"] == 0

    def test_wins_accumulate_correctly(self):
        # LAL wins games 1 and 2, enters game 3 with 2 wins in L10
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "GSW", 108, 102, "2024-01-03", 2024),
            ("LAL", "MIA", 105,  98, "2024-01-05", 2024),
        )
        result = compute_wins_rolling(games, window=10)
        assert result.iloc[2]["HOME_WINS_L10"] == 2

    def test_window_size_respected(self):
        # Build 12 LAL wins; entering game 13, window=10 → only 10 wins counted
        rows = []
        for i in range(12):
            date = f"2024-01-{i+1:02d}"
            rows.append(("LAL", "BOS", 110, 100, date, 2024))
        rows.append(("LAL", "MIA", 105, 98, "2024-01-13", 2024))
        games = _make_games(*rows)
        result = compute_wins_rolling(games, window=10)
        assert result.iloc[-1]["HOME_WINS_L10"] == 10

    def test_season_boundary_resets_wins(self):
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2023-04-01", 2023),
            ("LAL", "GSW", 108, 102, "2023-04-03", 2023),
            ("LAL", "MIA", 105,  98, "2024-01-05", 2024),  # new season
        )
        result = compute_wins_rolling(games, window=10)
        assert result.iloc[2]["HOME_WINS_L10"] == 0

    def test_no_data_leakage(self):
        # Wins entering game 2 should not include the result of game 2
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),  # LAL wins
            ("LAL", "GSW", 108, 102, "2024-01-03", 2024),
        )
        result = compute_wins_rolling(games, window=10)
        assert result.iloc[1]["HOME_WINS_L10"] == 1  # only 1 win before game 2

    def test_custom_window_column_name(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_wins_rolling(games, window=5)
        assert "HOME_WINS_L5" in result.columns
        assert "AWAY_WINS_L5" in result.columns

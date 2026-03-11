"""Tests for src/features/team/h2h.py."""

import pandas as pd
import pytest

from src.features.team.h2h import compute_h2h_records


def _make_games(*games: tuple) -> pd.DataFrame:
    rows = [
        {"HOME": h, "AWAY": a, "HOME_PTS": hp, "AWAY_PTS": ap, "DATE": d, "SEASON": s}
        for h, a, hp, ap, d, s in games
    ]
    return pd.DataFrame(rows)


class TestComputeH2HRecords:
    def test_output_has_required_columns(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_h2h_records(games)
        assert "HOME_REC" in result.columns
        assert "AWAY_REC" in result.columns

    def test_first_meeting_is_zero_for_both(self):
        games = _make_games(("LAL", "BOS", 110, 100, "2024-01-01", 2024))
        result = compute_h2h_records(games)
        assert result.iloc[0]["HOME_REC"] == 0.0
        assert result.iloc[0]["AWAY_REC"] == 0.0

    def test_after_home_win_home_rec_is_one(self):
        # LAL beats BOS in game 1; when they rematch, LAL's H2H record should be 1.0
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "BOS", 105,  98, "2024-02-01", 2024),
        )
        result = compute_h2h_records(games)
        assert result.iloc[1]["HOME_REC"] == 1.0
        assert result.iloc[1]["AWAY_REC"] == 0.0

    def test_after_away_win_away_rec_is_one(self):
        # BOS beats LAL in game 1; rematch should show BOS H2H record = 1.0
        games = _make_games(
            ("LAL", "BOS",  95, 110, "2024-01-01", 2024),  # BOS wins away
            ("LAL", "BOS", 105,  98, "2024-02-01", 2024),  # rematch
        )
        result = compute_h2h_records(games)
        assert result.iloc[1]["HOME_REC"] == 0.0
        assert result.iloc[1]["AWAY_REC"] == 1.0

    def test_records_sum_to_one_after_prior_meeting(self):
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "BOS", 105,  98, "2024-02-01", 2024),
        )
        result = compute_h2h_records(games)
        row = result.iloc[1]
        assert abs(row["HOME_REC"] + row["AWAY_REC"] - 1.0) < 1e-10

    def test_split_series_produces_correct_win_rate(self):
        # LAL wins 2, BOS wins 1 → entering game 4, LAL H2H = 2/3
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),  # LAL wins
            ("LAL", "BOS",  95, 108, "2024-01-15", 2024),  # BOS wins
            ("LAL", "BOS", 112, 105, "2024-02-01", 2024),  # LAL wins
            ("LAL", "BOS", 100,  98, "2024-03-01", 2024),  # check records
        )
        result = compute_h2h_records(games)
        assert abs(result.iloc[3]["HOME_REC"] - 2 / 3) < 1e-10
        assert abs(result.iloc[3]["AWAY_REC"] - 1 / 3) < 1e-10

    def test_new_season_resets_h2h(self):
        # LAL beats BOS in 2023; in 2024 their H2H should reset to 0.0
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2023-04-01", 2023),
            ("LAL", "BOS", 105,  98, "2024-01-15", 2024),
        )
        result = compute_h2h_records(games)
        assert result.iloc[1]["HOME_REC"] == 0.0
        assert result.iloc[1]["AWAY_REC"] == 0.0

    def test_h2h_is_opponent_specific(self):
        # LAL beating GSW does NOT affect LAL's H2H record vs BOS
        games = _make_games(
            ("LAL", "GSW", 110, 100, "2024-01-01", 2024),
            ("LAL", "BOS", 105,  98, "2024-01-03", 2024),
        )
        result = compute_h2h_records(games)
        bos_row = result[result["AWAY"] == "BOS"].iloc[0]
        assert bos_row["HOME_REC"] == 0.0  # no prior LAL vs BOS this season

    def test_no_data_leakage(self):
        # Records entering game 2 must reflect only game 1, not game 2 itself
        games = _make_games(
            ("LAL", "BOS", 110, 100, "2024-01-01", 2024),
            ("LAL", "BOS", 105,  98, "2024-02-01", 2024),
        )
        result = compute_h2h_records(games)
        # Game 2 entering record: LAL 1-0 (from game 1 only)
        assert result.iloc[1]["HOME_REC"] == 1.0

    def test_missing_columns_raises_value_error(self):
        bad_df = pd.DataFrame({"HOME": ["LAL"], "AWAY": ["BOS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_h2h_records(bad_df)

"""Tests for src/models/backtest.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import BacktestResult, _max_drawdown, _sharpe_ratio, run_game_backtest
from src.models.game_outcome import GameOutcomeModel, ALL_FEATURE_COLS, prepare_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n_per_season: int = 60, seasons=None) -> pd.DataFrame:
    """Synthetic feature DataFrame with outcome columns."""
    if seasons is None:
        seasons = list(range(2015, 2025))
    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            home_pts = int(rng.integers(90, 130))
            away_pts = int(rng.integers(90, 130))
            if home_pts == away_pts:
                away_pts += 1
            rows.append({
                "SEASON":            season,
                "DATE":              f"{season}-01-{(i % 28) + 1:02d}",
                "HOME":              "LAL",
                "AWAY":              "BOS",
                "HOME_PTS":          home_pts,
                "AWAY_PTS":          away_pts,
                "HOME_ELO":          1500.0 + rng.standard_normal() * 100,
                "AWAY_ELO":          1500.0 + rng.standard_normal() * 100,
                "HOME_B2B":          int(rng.integers(0, 2)),
                "AWAY_B2B":          int(rng.integers(0, 2)),
                "HOME_REST":         int(rng.integers(1, 7)),
                "AWAY_REST":         int(rng.integers(1, 7)),
                "HOME_STREAK":       int(rng.integers(0, 5)),
                "AWAY_STREAK":       int(rng.integers(0, 5)),
                "HOME_WINS_L10":     int(rng.integers(0, 10)),
                "AWAY_WINS_L10":     int(rng.integers(0, 10)),
                "HOME_REC":          float(rng.uniform(0, 1)),
                "AWAY_REC":          float(rng.uniform(0, 1)),
                "HOME_NET_RATING_L10": float(rng.standard_normal() * 5),
                "AWAY_NET_RATING_L10": float(rng.standard_normal() * 5),
                "PROJ_PACE_L10":     float(rng.uniform(95, 105)),
            })
    return pd.DataFrame(rows)


def _train_fast_model(feature_df: pd.DataFrame) -> GameOutcomeModel:
    """Train a minimal XGBoost model on synthetic data."""
    df = prepare_features(feature_df)
    train = df[df["SEASON"].isin(range(2015, 2023))]
    feat_cols = [c for c in ALL_FEATURE_COLS if c in train.columns]
    return GameOutcomeModel(
        params={"n_estimators": 10, "max_depth": 2}
    ).fit(train[feat_cols], train["LABEL"])


# ---------------------------------------------------------------------------
# run_game_backtest
# ---------------------------------------------------------------------------

class TestRunGameBacktest:
    def test_returns_backtest_result(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        assert isinstance(result, BacktestResult)

    def test_n_games_correct(self):
        n = 60
        df = _make_feature_df(n_per_season=n, seasons=list(range(2015, 2025)))
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        assert result.n_games == n

    def test_win_rate_in_zero_one(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024, edge_threshold=0.0)
        assert 0.0 <= result.win_rate <= 1.0

    def test_zero_bets_when_threshold_too_high(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        # No model can have > 100% edge, so threshold=1.0 yields 0 bets
        result = run_game_backtest(df, model, test_seasons=2024, edge_threshold=1.0)
        assert result.n_bets == 0
        assert np.isnan(result.win_rate)
        assert np.isnan(result.roi)

    def test_all_bets_when_threshold_zero(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024, edge_threshold=0.0)
        assert result.n_bets == result.n_games

    def test_brier_score_in_range(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        assert 0.0 <= result.brier_score <= 1.0

    def test_max_drawdown_non_negative(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        assert result.max_drawdown >= 0.0

    def test_calibration_is_dataframe(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        assert isinstance(result.calibration, pd.DataFrame)

    def test_picks_df_columns(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        for col in ("HOME", "AWAY", "DATE", "HOME_WIN_PROB", "EDGE", "BET", "WIN", "PNL"):
            assert col in result.picks_df.columns

    def test_missing_test_season_raises(self):
        df = _make_feature_df(seasons=[2022, 2023])
        model = _train_fast_model(_make_feature_df())
        with pytest.raises(ValueError, match="No data found"):
            run_game_backtest(df, model, test_seasons=2024)

    def test_summary_returns_string(self):
        df = _make_feature_df()
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=2024)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Win rate" in summary

    def test_multiple_test_seasons(self):
        df = _make_feature_df(n_per_season=30)
        model = _train_fast_model(df)
        result = run_game_backtest(df, model, test_seasons=[2023, 2024])
        assert result.n_games == 60   # 30 per season


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_zero_on_all_wins(self):
        pnl = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert _max_drawdown(pnl) == 0.0

    def test_simple_drawdown(self):
        # Peak at 3, then drops to 1 → drawdown = 2
        pnl = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0])
        assert _max_drawdown(pnl) == 2.0

    def test_empty_series(self):
        assert _max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestSharpeRatio:
    def test_positive_sharpe_for_positive_mean(self):
        # Varying positive values — Sharpe should be > 0
        rng = np.random.default_rng(0)
        daily_pnl = pd.Series(rng.normal(loc=1.0, scale=0.5, size=82))
        assert _sharpe_ratio(daily_pnl) > 0.0

    def test_zero_sharpe_for_zero_std(self):
        # Constant P&L — std = 0 → Sharpe = 0 (not inf)
        daily_pnl = pd.Series([1.0] * 10)
        # All same → std=0 → we guard against divide-by-zero
        # Actually np.std([1,1,...]) = 0, so we return 0
        result = _sharpe_ratio(daily_pnl)
        assert result == 0.0

    def test_empty_series(self):
        assert _sharpe_ratio(pd.Series([], dtype=float)) == 0.0

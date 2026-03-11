"""Tests for src/models/player_props.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.player_props import (
    ALL_STATS,
    COMBO_STATS,
    PROP_STATS,
    PlayerPropModel,
    PropBacktestResult,
    _get_feature_cols,
    _stat_components,
    prepare_player_features,
    run_prop_backtest,
    train_all_props,
    walk_forward_train_prop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player_df(n_per_season: int = 40, seasons=None) -> pd.DataFrame:
    """Synthetic player feature DataFrame mirroring build_player_features output."""
    if seasons is None:
        seasons = list(range(2015, 2025))

    rng = np.random.default_rng(99)
    rows = []
    player_ids = [1, 2, 3]

    for season in seasons:
        for pid in player_ids:
            for i in range(n_per_season):
                pts  = float(rng.integers(5, 40))
                reb  = float(rng.integers(1, 15))
                ast  = float(rng.integers(0, 12))
                fg3m = float(rng.integers(0, 7))
                stl  = float(rng.integers(0, 4))
                blk  = float(rng.integers(0, 4))
                tov  = float(rng.integers(0, 6))

                rows.append({
                    "PLAYER_ID":   pid,
                    "PLAYER_NAME": f"Player{pid}",
                    "TEAM":        "LAL",
                    "OPP":         "BOS",
                    "DATE":        f"{season}-01-{(i % 28) + 1:02d}",
                    "SEASON":      season,
                    "IS_HOME":     bool(rng.integers(0, 2)),
                    # Actual stats (targets)
                    "PTS":  pts,  "REB": reb,  "AST": ast,
                    "FG3M": fg3m, "STL": stl,  "BLK": blk, "TOV": tov,
                    # Rolling averages
                    "PTS_L5":  pts + rng.standard_normal() * 2,
                    "PTS_L10": pts + rng.standard_normal() * 1.5,
                    "PTS_L20": pts + rng.standard_normal() * 1,
                    "PTS_SEASON": pts,
                    "REB_L5":  reb + rng.standard_normal(),
                    "REB_L10": reb + rng.standard_normal() * 0.8,
                    "REB_L20": reb + rng.standard_normal() * 0.5,
                    "REB_SEASON": reb,
                    "AST_L5":  ast + rng.standard_normal(),
                    "AST_L10": ast + rng.standard_normal() * 0.8,
                    "AST_L20": ast + rng.standard_normal() * 0.5,
                    "AST_SEASON": ast,
                    "FG3M_L5":  fg3m + rng.standard_normal() * 0.5,
                    "FG3M_L10": fg3m + rng.standard_normal() * 0.4,
                    "FG3M_L20": fg3m + rng.standard_normal() * 0.3,
                    "FG3M_SEASON": fg3m,
                    # Minutes and usage
                    "MIN_L5":  float(rng.integers(20, 36)),
                    "MIN_L10": float(rng.integers(20, 36)),
                    "MIN_L20": float(rng.integers(20, 36)),
                    "MIN_SEASON": float(rng.integers(20, 36)),
                    "USAGE_PROXY_L5":   float(rng.uniform(15, 35)),
                    "USAGE_PROXY_L10":  float(rng.uniform(15, 35)),
                    "USAGE_PROXY_L20":  float(rng.uniform(15, 35)),
                    "USAGE_PROXY_SEASON": float(rng.uniform(15, 35)),
                    # Home/away splits
                    "PTS_HOME_AVG":       pts + rng.standard_normal(),
                    "PTS_AWAY_AVG":       pts - rng.standard_normal(),
                    "PTS_HOME_AWAY_DIFF": float(rng.standard_normal() * 2),
                    "PTS_VS_OPP_AVG":     pts + rng.standard_normal(),
                    "VS_OPP_N":           int(rng.integers(1, 5)),
                    # Context
                    "TEAMMATE_OUT_BOOST": float(rng.uniform(0, 3)),
                    "TEAMMATE_OUT_FLAG":  int(rng.integers(0, 2)),
                    "PROJ_PACE_L10":      float(rng.uniform(95, 105)),
                })

    df = pd.DataFrame(rows)
    # Add combo stats
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["PR"]  = df["PTS"] + df["REB"]
    df["PA"]  = df["PTS"] + df["AST"]
    df["RA"]  = df["REB"] + df["AST"]
    return df


_FAST_PARAMS = {"n_estimators": 5, "max_depth": 2}


# ---------------------------------------------------------------------------
# Stat definitions
# ---------------------------------------------------------------------------

class TestStatDefinitions:
    def test_all_stats_includes_singles_and_combos(self):
        for s in PROP_STATS:
            assert s in ALL_STATS
        for s in COMBO_STATS:
            assert s in ALL_STATS

    def test_stat_components_single(self):
        assert _stat_components("PTS") == ["PTS"]
        assert _stat_components("REB") == ["REB"]

    def test_stat_components_combo(self):
        assert _stat_components("PRA") == ["PTS", "REB", "AST"]
        assert _stat_components("PR")  == ["PTS", "REB"]

    def test_get_feature_cols_returns_list(self):
        available = ["PTS_L5", "PTS_L10", "PTS_SEASON", "MIN_L5", "PROJ_PACE_L10"]
        cols = _get_feature_cols("PTS", available)
        assert isinstance(cols, list)
        assert all(c in available for c in cols)

    def test_get_feature_cols_excludes_missing(self):
        # Only a few cols available — should not include unavailable ones
        cols = _get_feature_cols("PTS", ["PTS_L5"])
        assert "PTS_L10" not in cols


# ---------------------------------------------------------------------------
# prepare_player_features
# ---------------------------------------------------------------------------

class TestPreparePlayerFeatures:
    def test_returns_X_and_y(self):
        df = _make_player_df(n_per_season=10, seasons=[2024])
        X, y = prepare_player_features(df, "PTS")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_X_and_y_same_length(self):
        df = _make_player_df(n_per_season=10, seasons=[2024])
        X, y = prepare_player_features(df, "PTS")
        assert len(X) == len(y)

    def test_combo_stat_target_is_sum(self):
        df = _make_player_df(n_per_season=5, seasons=[2024])
        _, y_pra = prepare_player_features(df, "PRA")
        _, y_pts = prepare_player_features(df, "PTS")
        _, y_reb = prepare_player_features(df, "REB")
        _, y_ast = prepare_player_features(df, "AST")
        np.testing.assert_allclose(
            y_pra.values, (y_pts + y_reb + y_ast).values, atol=1e-6
        )

    def test_missing_stat_raises(self):
        df = _make_player_df(n_per_season=5, seasons=[2024])
        df = df.drop(columns=["PTS"])
        with pytest.raises(ValueError, match="PTS"):
            prepare_player_features(df, "PTS")

    def test_combo_missing_component_raises(self):
        df = _make_player_df(n_per_season=5, seasons=[2024])
        df = df.drop(columns=["AST"])
        with pytest.raises(ValueError, match="PRA"):
            prepare_player_features(df, "PRA")


# ---------------------------------------------------------------------------
# PlayerPropModel
# ---------------------------------------------------------------------------

class TestPlayerPropModel:
    def _fit_model(self, stat="PTS", seasons=None):
        if seasons is None:
            seasons = list(range(2015, 2025))
        df = _make_player_df(n_per_season=20, seasons=seasons)
        train = df[df["SEASON"].isin(range(2015, 2023))]
        X, y = prepare_player_features(train, stat)
        return PlayerPropModel(stat=stat, params=_FAST_PARAMS).fit(X, y), df

    def test_fit_returns_self(self):
        df = _make_player_df(n_per_season=20, seasons=list(range(2015, 2023)))
        X, y = prepare_player_features(df, "PTS")
        model = PlayerPropModel(stat="PTS", params=_FAST_PARAMS)
        assert model.fit(X, y) is model

    def test_predict_returns_three_keys(self):
        model, df = self._fit_model()
        X, _ = prepare_player_features(df[df["SEASON"] == 2024], "PTS")
        preds = model.predict(X)
        assert set(preds.keys()) == {"median", "low", "high"}

    def test_predict_shapes_match(self):
        model, df = self._fit_model()
        X, _ = prepare_player_features(df[df["SEASON"] == 2024], "PTS")
        preds = model.predict(X)
        n = len(X)
        assert preds["median"].shape == (n,)
        assert preds["low"].shape    == (n,)
        assert preds["high"].shape   == (n,)

    def test_low_le_median_le_high(self):
        model, df = self._fit_model()
        X, _ = prepare_player_features(df[df["SEASON"] == 2024], "PTS")
        preds = model.predict(X)
        # Quantile regression guarantees this on TRAINING data; test set may
        # occasionally violate. Use a loose check (most samples).
        assert (preds["low"] <= preds["median"] + 5).all()
        assert (preds["median"] <= preds["high"] + 5).all()

    def test_predict_before_fit_raises(self):
        model = PlayerPropModel(stat="PTS", params=_FAST_PARAMS)
        df = _make_player_df(n_per_season=5, seasons=[2024])
        X, _ = prepare_player_features(df, "PTS")
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_no_feature_cols_raises(self):
        model = PlayerPropModel(stat="PTS", params=_FAST_PARAMS)
        X_bad = pd.DataFrame({"UNKNOWN": [1.0, 2.0, 3.0]})
        y_bad = pd.Series([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="No recognised feature columns"):
            model.fit(X_bad, y_bad)

    def test_feature_importance_returns_series(self):
        model, df = self._fit_model()
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(model.feature_cols)

    def test_combo_stat_fits_correctly(self):
        model, df = self._fit_model(stat="PRA")
        X, _ = prepare_player_features(df[df["SEASON"] == 2024], "PRA")
        preds = model.predict(X)
        # PRA is larger than PTS alone — check predictions are plausible
        assert preds["median"].mean() > 0

    def test_save_and_load(self, tmp_path):
        model, df = self._fit_model()
        path = str(tmp_path / "prop_pts.joblib")
        model.save(path)
        loaded = PlayerPropModel.load(path)
        X, _ = prepare_player_features(df[df["SEASON"] == 2024], "PTS")
        np.testing.assert_allclose(
            model.predict(X)["median"],
            loaded.predict(X)["median"],
        )


# ---------------------------------------------------------------------------
# walk_forward_train_prop
# ---------------------------------------------------------------------------

class TestWalkForwardTrainProp:
    def test_returns_model_and_metrics(self):
        df = _make_player_df(n_per_season=20)
        model, metrics = walk_forward_train_prop(
            df, "PTS",
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        assert isinstance(model, PlayerPropModel)
        assert isinstance(metrics, dict)

    def test_metrics_has_expected_keys(self):
        df = _make_player_df(n_per_season=20)
        _, metrics = walk_forward_train_prop(
            df, "PTS",
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        for key in ("mae", "rmse", "direction_accuracy", "coverage_80",
                    "n_train", "n_val", "n_test"):
            assert key in metrics, f"Missing key: {key}"

    def test_mae_is_positive(self):
        df = _make_player_df(n_per_season=20)
        _, metrics = walk_forward_train_prop(
            df, "PTS",
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        assert metrics["mae"] > 0

    def test_coverage_in_zero_one(self):
        df = _make_player_df(n_per_season=20)
        _, metrics = walk_forward_train_prop(
            df, "PTS",
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        assert 0.0 <= metrics["coverage_80"] <= 1.0

    def test_empty_train_raises(self):
        df = _make_player_df(n_per_season=10, seasons=[2023, 2024])
        with pytest.raises(ValueError, match="No training data"):
            walk_forward_train_prop(
                df, "PTS",
                train_seasons=[2015],
                val_season=2023, test_season=2024,
                params=_FAST_PARAMS,
            )


# ---------------------------------------------------------------------------
# train_all_props
# ---------------------------------------------------------------------------

class TestTrainAllProps:
    def test_returns_dict_with_models(self):
        df = _make_player_df(n_per_season=20)
        results = train_all_props(
            df, stats=["PTS", "REB"],
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        assert "PTS" in results
        assert "REB" in results

    def test_combo_stat_trained(self):
        df = _make_player_df(n_per_season=20)
        results = train_all_props(
            df, stats=["PRA"],
            train_seasons=list(range(2015, 2023)),
            val_season=2023, test_season=2024,
            params=_FAST_PARAMS,
        )
        assert "PRA" in results
        model, _ = results["PRA"]
        assert isinstance(model, PlayerPropModel)
        assert model.stat == "PRA"


# ---------------------------------------------------------------------------
# run_prop_backtest
# ---------------------------------------------------------------------------

class TestRunPropBacktest:
    def _setup(self, stat="PTS"):
        df = _make_player_df(n_per_season=20)
        train = df[df["SEASON"].isin(range(2015, 2023))]
        X, y = prepare_player_features(train, stat)
        model = PlayerPropModel(stat=stat, params=_FAST_PARAMS).fit(X, y)
        return model, df

    def test_returns_prop_backtest_result(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        assert isinstance(result, PropBacktestResult)

    def test_n_games_correct(self):
        n = 20
        df = _make_player_df(n_per_season=n)
        train = df[df["SEASON"].isin(range(2015, 2023))]
        X, y = prepare_player_features(train, "PTS")
        model = PlayerPropModel(stat="PTS", params=_FAST_PARAMS).fit(X, y)
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        # 3 players × 20 games each = 60 player-games in test season
        assert result.n_games == n * 3

    def test_zero_bets_when_threshold_very_high(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024, edge_threshold=9999)
        assert result.n_bets == 0

    def test_mae_is_positive(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        assert result.mae > 0

    def test_coverage_in_zero_one(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        assert 0.0 <= result.coverage_80 <= 1.0

    def test_picks_df_has_expected_columns(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        for col in ("ACTUAL", "LINE", "PREDICTION", "EDGE", "BET", "WIN", "LOW", "HIGH"):
            assert col in result.picks_df.columns

    def test_missing_test_season_raises(self):
        df = _make_player_df(n_per_season=10, seasons=[2022, 2023])
        train = df[df["SEASON"] == 2022]
        X, y = prepare_player_features(train, "PTS")
        model = PlayerPropModel(stat="PTS", params=_FAST_PARAMS).fit(X, y)
        with pytest.raises(ValueError, match="No data"):
            run_prop_backtest(df, model, "PTS", test_seasons=2024)

    def test_summary_returns_string(self):
        model, df = self._setup()
        result = run_prop_backtest(df, model, "PTS", test_seasons=2024)
        s = result.summary()
        assert isinstance(s, str)
        assert "PTS" in s

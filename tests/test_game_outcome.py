"""Tests for src/models/game_outcome.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.game_outcome import (
    ALL_FEATURE_COLS,
    GameOutcomeModel,
    prepare_features,
    walk_forward_train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n_per_season: int = 60, seasons=None) -> pd.DataFrame:
    """Synthetic feature DataFrame that matches the expected pipeline output."""
    if seasons is None:
        seasons = list(range(2015, 2025))   # 2015-2024

    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            home_pts = int(rng.integers(90, 130))
            away_pts = int(rng.integers(90, 130))
            # Avoid ties
            if home_pts == away_pts:
                away_pts += 1
            rows.append({
                "SEASON":    season,
                "DATE":      f"{season}-01-{(i % 28) + 1:02d}",
                "HOME":      "LAL",
                "AWAY":      "BOS",
                "HOME_PTS":  home_pts,
                "AWAY_PTS":  away_pts,
                # ELO features
                "HOME_ELO":  1500.0 + rng.standard_normal() * 100,
                "AWAY_ELO":  1500.0 + rng.standard_normal() * 100,
                # Schedule features
                "HOME_B2B":  int(rng.integers(0, 2)),
                "AWAY_B2B":  int(rng.integers(0, 2)),
                "HOME_REST": int(rng.integers(1, 7)),
                "AWAY_REST": int(rng.integers(1, 7)),
                # Form features
                "HOME_STREAK":   int(rng.integers(0, 5)),
                "AWAY_STREAK":   int(rng.integers(0, 5)),
                "HOME_WINS_L10": int(rng.integers(0, 10)),
                "AWAY_WINS_L10": int(rng.integers(0, 10)),
                # H2H
                "HOME_REC": float(rng.uniform(0, 1)),
                "AWAY_REC": float(rng.uniform(0, 1)),
                # A couple of rolling stats so XGBoost has real features
                "HOME_NET_RATING_L10": float(rng.standard_normal() * 5),
                "AWAY_NET_RATING_L10": float(rng.standard_normal() * 5),
                "PROJ_PACE_L10":       float(rng.uniform(95, 105)),
            })
    return pd.DataFrame(rows)


_FAST_PARAMS = {"n_estimators": 10, "max_depth": 2}


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_adds_label(self):
        df = _make_feature_df(n_per_season=10, seasons=[2024])
        result = prepare_features(df)
        assert "LABEL" in result.columns

    def test_label_is_binary(self):
        df = _make_feature_df(n_per_season=20, seasons=[2024])
        result = prepare_features(df)
        assert set(result["LABEL"].unique()).issubset({0, 1})

    def test_label_home_win_is_one(self):
        df = pd.DataFrame([{
            "HOME_PTS": 110, "AWAY_PTS": 100,
            "HOME_ELO": 1500, "AWAY_ELO": 1500,
        }])
        result = prepare_features(df)
        assert result.iloc[0]["LABEL"] == 1

    def test_label_away_win_is_zero(self):
        df = pd.DataFrame([{
            "HOME_PTS": 95, "AWAY_PTS": 105,
            "HOME_ELO": 1500, "AWAY_ELO": 1500,
        }])
        result = prepare_features(df)
        assert result.iloc[0]["LABEL"] == 0

    def test_adds_elo_diff(self):
        df = pd.DataFrame([{
            "HOME_PTS": 110, "AWAY_PTS": 100,
            "HOME_ELO": 1550.0, "AWAY_ELO": 1480.0,
        }])
        result = prepare_features(df)
        assert "ELO_DIFF" in result.columns
        assert abs(result.iloc[0]["ELO_DIFF"] - 70.0) < 1e-6

    def test_drops_ties(self):
        df = pd.DataFrame([
            {"HOME_PTS": 100, "AWAY_PTS": 100, "HOME_ELO": 1500, "AWAY_ELO": 1500},
            {"HOME_PTS": 110, "AWAY_PTS": 100, "HOME_ELO": 1500, "AWAY_ELO": 1500},
        ])
        result = prepare_features(df)
        assert len(result) == 1

    def test_no_elo_still_works(self):
        df = pd.DataFrame([{"HOME_PTS": 110, "AWAY_PTS": 100}])
        result = prepare_features(df)
        assert "LABEL" in result.columns
        assert "ELO_DIFF" not in result.columns


# ---------------------------------------------------------------------------
# GameOutcomeModel
# ---------------------------------------------------------------------------

class TestGameOutcomeModel:
    def _X_y(self, n=100):
        df = _make_feature_df(n_per_season=n, seasons=[2022])
        df = prepare_features(df)
        feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        return df[feat_cols], df["LABEL"]

    def test_fit_returns_self(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS)
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_shape(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_predict_proba_in_zero_one(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        probs = model.predict_proba(X)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_probs_sum_to_one(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importance_returns_series(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(model.feature_cols)

    def test_feature_importance_after_calibration(self):
        # After walk_forward_train with calibrate=True, model.model is a
        # CalibratedModel wrapper. feature_importance() must still work.
        from src.models.calibration import calibrate_model
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        model.model = calibrate_model(model.model, X, y)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(model.feature_cols)

    def test_predict_before_fit_raises(self):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(X)

    def test_no_recognised_features_raises(self):
        model = GameOutcomeModel(params=_FAST_PARAMS)
        X_bad = pd.DataFrame({"UNKNOWN_COL": [1, 2, 3]})
        y_bad = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="No recognised feature columns"):
            model.fit(X_bad, y_bad)

    def test_save_and_load(self, tmp_path):
        X, y = self._X_y()
        model = GameOutcomeModel(params=_FAST_PARAMS).fit(X, y)
        path = str(tmp_path / "model.joblib")
        model.save(path)
        loaded = GameOutcomeModel.load(path)
        np.testing.assert_allclose(
            model.predict_proba(X),
            loaded.predict_proba(X),
        )


# ---------------------------------------------------------------------------
# walk_forward_train
# ---------------------------------------------------------------------------

class TestWalkForwardTrain:
    def test_returns_model_and_metrics(self):
        df = _make_feature_df(n_per_season=30)
        model, metrics = walk_forward_train(
            df,
            train_seasons=list(range(2015, 2023)),
            val_season=2023,
            test_season=2024,
            calibrate=False,
            params=_FAST_PARAMS,
        )
        assert isinstance(model, GameOutcomeModel)
        assert isinstance(metrics, dict)

    def test_metrics_has_expected_keys(self):
        df = _make_feature_df(n_per_season=30)
        _, metrics = walk_forward_train(
            df,
            train_seasons=list(range(2015, 2023)),
            val_season=2023,
            test_season=2024,
            calibrate=False,
            params=_FAST_PARAMS,
        )
        for key in ("val_accuracy", "val_brier", "val_logloss",
                    "test_accuracy", "test_brier", "test_logloss",
                    "n_train", "n_val", "n_test", "n_features"):
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_in_range(self):
        df = _make_feature_df(n_per_season=30)
        _, metrics = walk_forward_train(
            df,
            train_seasons=list(range(2015, 2023)),
            val_season=2023,
            test_season=2024,
            calibrate=False,
            params=_FAST_PARAMS,
        )
        assert 0.0 <= metrics["test_accuracy"] <= 1.0

    def test_sample_counts_correct(self):
        df = _make_feature_df(n_per_season=30)
        _, metrics = walk_forward_train(
            df,
            train_seasons=list(range(2015, 2023)),
            val_season=2023,
            test_season=2024,
            calibrate=False,
            params=_FAST_PARAMS,
        )
        assert metrics["n_train"] == 8 * 30
        assert metrics["n_val"]   == 30
        assert metrics["n_test"]  == 30

    def test_empty_train_raises(self):
        df = _make_feature_df(n_per_season=30, seasons=[2023, 2024])
        with pytest.raises(ValueError, match="No training data"):
            walk_forward_train(
                df,
                train_seasons=[2015],   # no 2015 data
                val_season=2023,
                test_season=2024,
                calibrate=False,
                params=_FAST_PARAMS,
            )

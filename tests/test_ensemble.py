"""Tests for src/models/ensemble.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.ensemble import (
    EloLogitModel,
    EnsembleModel,
    ELO_FEATURE_COLS,
    FF_PACE_FEATURE_COLS,
    FourFactorLGBModel,
)
from src.models.game_outcome import ALL_FEATURE_COLS, GameOutcomeModel, prepare_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy(n=120):
    """Minimal feature DataFrame with a few key columns."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "HOME_ELO":            1500 + rng.standard_normal(n) * 100,
        "AWAY_ELO":            1500 + rng.standard_normal(n) * 100,
        "ELO_DIFF":            rng.standard_normal(n) * 120,
        "HOME_B2B":            rng.integers(0, 2, n).astype(float),
        "AWAY_B2B":            rng.integers(0, 2, n).astype(float),
        "HOME_REST":           rng.integers(1, 7, n).astype(float),
        "AWAY_REST":           rng.integers(1, 7, n).astype(float),
        "HOME_STREAK":         rng.integers(0, 5, n).astype(float),
        "AWAY_STREAK":         rng.integers(0, 5, n).astype(float),
        "HOME_WINS_L10":       rng.integers(0, 10, n).astype(float),
        "AWAY_WINS_L10":       rng.integers(0, 10, n).astype(float),
        "HOME_REC":            rng.uniform(0, 1, n),
        "AWAY_REC":            rng.uniform(0, 1, n),
        "HOME_NET_RATING_L10": rng.standard_normal(n) * 5,
        "AWAY_NET_RATING_L10": rng.standard_normal(n) * 5,
        "PROJ_PACE_L10":       rng.uniform(95, 105, n),
        "HOME_EFG_PCT_L10":    rng.uniform(0.45, 0.60, n),
        "AWAY_EFG_PCT_L10":    rng.uniform(0.45, 0.60, n),
        "HOME_FTR_L10":        rng.uniform(0.15, 0.35, n),
        "AWAY_FTR_L10":        rng.uniform(0.15, 0.35, n),
        "HOME_TOV_PCT_L10":    rng.uniform(10, 18, n),
        "AWAY_TOV_PCT_L10":    rng.uniform(10, 18, n),
        "HOME_ORB_PCT_L10":    rng.uniform(20, 35, n),
        "AWAY_ORB_PCT_L10":    rng.uniform(20, 35, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="LABEL")
    return df, y


_FAST_XGB = {"n_estimators": 10, "max_depth": 2}
_FAST_LGB = {"n_estimators": 10, "max_depth": 2}


# ---------------------------------------------------------------------------
# EloLogitModel
# ---------------------------------------------------------------------------

class TestEloLogitModel:
    def test_fit_and_predict(self):
        X, y = _make_xy()
        model = EloLogitModel().fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_probs_in_zero_one(self):
        X, y = _make_xy()
        model = EloLogitModel().fit(X, y)
        probs = model.predict_proba(X)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_only_uses_elo_cols(self):
        X, y = _make_xy()
        model = EloLogitModel().fit(X, y)
        assert all(c in ELO_FEATURE_COLS for c in model.feature_cols)


# ---------------------------------------------------------------------------
# FourFactorLGBModel
# ---------------------------------------------------------------------------

class TestFourFactorLGBModel:
    def test_fit_and_predict(self):
        X, y = _make_xy()
        model = FourFactorLGBModel(params=_FAST_LGB).fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_probs_in_zero_one(self):
        X, y = _make_xy()
        model = FourFactorLGBModel(params=_FAST_LGB).fit(X, y)
        probs = model.predict_proba(X)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_only_uses_ff_pace_cols(self):
        X, y = _make_xy()
        model = FourFactorLGBModel(params=_FAST_LGB).fit(X, y)
        assert all(c in FF_PACE_FEATURE_COLS for c in model.feature_cols)


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    def _build_ensemble(self, X, y):
        ma = GameOutcomeModel(params=_FAST_XGB).fit(X, y)
        mb = EloLogitModel().fit(X, y)
        mc = FourFactorLGBModel(params=_FAST_LGB).fit(X, y)
        return EnsembleModel(
            models=[ma, mb, mc],
            feature_col_sets=[ALL_FEATURE_COLS, ELO_FEATURE_COLS, FF_PACE_FEATURE_COLS],
        )

    def test_mismatched_lengths_raises(self):
        X, y = _make_xy()
        mb = EloLogitModel().fit(X, y)
        with pytest.raises(ValueError, match="same length"):
            EnsembleModel(models=[mb], feature_col_sets=[ELO_FEATURE_COLS, FF_PACE_FEATURE_COLS])

    def test_predict_proba_shape(self):
        X, y = _make_xy()
        ensemble = self._build_ensemble(X, y)
        probs = ensemble.predict_proba(X)
        assert probs.shape == (len(X),)

    def test_predict_proba_in_zero_one(self):
        X, y = _make_xy()
        ensemble = self._build_ensemble(X, y)
        probs = ensemble.predict_proba(X)
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_equal_weights_by_default(self):
        X, y = _make_xy()
        mb = EloLogitModel().fit(X, y)
        ensemble = EnsembleModel(
            models=[mb, mb],
            feature_col_sets=[ELO_FEATURE_COLS, ELO_FEATURE_COLS],
        )
        assert abs(ensemble.weights[0] - 0.5) < 1e-9
        assert abs(ensemble.weights[1] - 0.5) < 1e-9

    def test_custom_weights_respected(self):
        X, y = _make_xy()
        mb = EloLogitModel().fit(X, y)
        ensemble = EnsembleModel(
            models=[mb, mb],
            feature_col_sets=[ELO_FEATURE_COLS, ELO_FEATURE_COLS],
            weights=[0.7, 0.3],
        )
        assert abs(ensemble.weights[0] - 0.7) < 1e-9

    def test_tune_weights_returns_list(self):
        X, y = _make_xy(n=120)
        ensemble = self._build_ensemble(X, y)
        weights = ensemble.tune_weights(X, y)
        assert isinstance(weights, list)
        assert len(weights) == 3

    def test_tune_weights_sum_to_one(self):
        X, y = _make_xy(n=120)
        ensemble = self._build_ensemble(X, y)
        ensemble.tune_weights(X, y)
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6

    def test_tune_weights_non_negative(self):
        X, y = _make_xy(n=120)
        ensemble = self._build_ensemble(X, y)
        ensemble.tune_weights(X, y)
        assert all(w >= 0.0 for w in ensemble.weights)

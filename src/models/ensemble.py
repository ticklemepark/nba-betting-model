"""Ensemble model combining three game outcome classifiers.

Component models:
    A — GameOutcomeModel (XGBoost) on the full feature set
    B — EloLogitModel (Logistic Regression) on ELO features only
        Strong parametric baseline; hard to beat on ELO-dominated games.
    C — FourFactorLGBModel (LightGBM) on Four Factors + pace features only
        Captures current-form efficiency signals that ELO misses.

Combination:
    Weighted average of each model's predicted P(home win). Weights are
    tuned on the validation set by fitting a non-negative stacking
    logistic regression (sklearn only — no scipy needed).

Usage:
    from src.models.ensemble import (
        EnsembleModel, EloLogitModel, FourFactorLGBModel
    )
    from src.models.game_outcome import GameOutcomeModel, ALL_FEATURE_COLS

    model_a = GameOutcomeModel().fit(X_train, y_train)
    model_b = EloLogitModel().fit(X_train, y_train)
    model_c = FourFactorLGBModel().fit(X_train, y_train)

    ensemble = EnsembleModel(
        models=[model_a, model_b, model_c],
        feature_col_sets=[ALL_FEATURE_COLS, ELO_FEATURE_COLS, FF_PACE_FEATURE_COLS],
    )
    ensemble.tune_weights(X_val, y_val)
    probs = ensemble.predict_proba(X_test)   # shape (n,) — P(home win)
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.models.game_outcome import ALL_FEATURE_COLS, _ELO_COLS, _FOUR_FACTOR_COLS, _PACE_COLS

log = logging.getLogger(__name__)

# Feature subsets for each component model
ELO_FEATURE_COLS     = _ELO_COLS
FF_PACE_FEATURE_COLS = _FOUR_FACTOR_COLS + _PACE_COLS

_DEFAULT_LGB_PARAMS: dict = {
    "n_estimators":       300,
    "max_depth":          4,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_samples":  10,
    "reg_alpha":          0.1,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EnsembleModel:
    """Weighted average ensemble of game outcome classifiers.

    Each component model predicts P(home win) independently. The final
    prediction is a weighted average, with weights optimised on the
    validation set.
    """

    def __init__(
        self,
        models: list,
        feature_col_sets: list[list[str]],
        weights: list[float] | None = None,
    ):
        """
        Args:
            models:           Fitted model objects with predict_proba(X).
            feature_col_sets: Feature column list for each model (same order).
            weights:          Initial weights. Defaults to equal weighting.

        Raises:
            ValueError: If models and feature_col_sets differ in length.
        """
        if len(models) != len(feature_col_sets):
            raise ValueError(
                "models and feature_col_sets must have the same length: "
                f"{len(models)} vs {len(feature_col_sets)}"
            )
        self.models = models
        self.feature_col_sets = feature_col_sets
        n = len(models)
        self.weights: list[float] = list(weights) if weights else [1.0 / n] * n

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average P(home win) for each row. Shape: (n,)."""
        probs = self._component_probs(X)
        return np.average(probs, weights=self.weights, axis=0)

    def tune_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> list[float]:
        """Optimise ensemble weights on the validation set.

        Fits a non-negative stacking logistic regression where each
        component model's probability is a feature. Coefficients are
        clipped to [0, ∞) and normalised to sum to 1.

        Args:
            X_val: Validation feature matrix (same columns as training).
            y_val: Validation labels.

        Returns:
            Updated self.weights list.
        """
        probs = self._component_probs(X_val)          # (n_models, n_samples)
        prob_matrix = np.column_stack(probs)           # (n_samples, n_models)

        stacker = LogisticRegression(
            fit_intercept=False, C=1.0, max_iter=500, random_state=42
        )
        stacker.fit(prob_matrix, y_val)

        raw = stacker.coef_.flatten()
        clipped = np.clip(raw, 0.0, None)
        total = clipped.sum()

        if total <= 0:
            log.warning(
                "All stacker coefficients non-positive — keeping equal weights."
            )
            return self.weights

        self.weights = (clipped / total).tolist()

        equal_brier = brier_score_loss(y_val, np.mean(probs, axis=0))
        tuned_brier = brier_score_loss(y_val, self.predict_proba(X_val))
        log.info(
            "Ensemble weights tuned: %s  |  val Brier equal=%.4f → tuned=%.4f",
            [f"{w:.3f}" for w in self.weights],
            equal_brier,
            tuned_brier,
        )
        return self.weights

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _component_probs(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Return list of (n_samples,) probability arrays, one per model."""
        results = []
        for model, cols in zip(self.models, self.feature_col_sets):
            available = [c for c in cols if c in X.columns]
            X_sub = X[available].fillna(0.0)
            p = model.predict_proba(X_sub)
            if p.ndim == 2:
                p = p[:, 1]
            results.append(p)
        return results


class EloLogitModel:
    """Logistic regression on ELO features — fast, interpretable baseline.

    Uses only HOME_ELO, AWAY_ELO, and ELO_DIFF. Serves as Model B in the
    ensemble to provide a well-calibrated parametric baseline.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EloLogitModel":
        self.feature_cols = [c for c in ELO_FEATURE_COLS if c in X.columns]
        self.model.fit(X[self.feature_cols].fillna(0.0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols].fillna(0.0))


class FourFactorLGBModel:
    """LightGBM on Four Factors + pace features — captures current-form signals.

    Complements ELO (which is slow-moving) with efficiency metrics from the
    last 5/10/20 games. Serves as Model C in the ensemble.
    """

    def __init__(self, params: dict | None = None):
        self.params = {**_DEFAULT_LGB_PARAMS, **(params or {})}
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "FourFactorLGBModel":
        self.feature_cols = [c for c in FF_PACE_FEATURE_COLS if c in X.columns]
        X_fit = X[self.feature_cols].fillna(0.0)

        if X_val is not None and y_val is not None:
            X_vf = X_val[self.feature_cols].fillna(0.0)
            self.model.fit(X_fit, y, eval_set=[(X_vf, y_val)])
        else:
            self.model.fit(X_fit, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols].fillna(0.0))

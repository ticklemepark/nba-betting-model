"""Game outcome model: predicts the probability of the home team winning.

Architecture: XGBoost classifier (primary) trained with walk-forward validation.
  - Train:    seasons 2015–2022  (8 seasons, ~9,800 games)
  - Validate: 2023               (calibration + hyperparameter checks)
  - Test:     2024               (held-out, never touched during development)

All features are pre-game (zero data leakage). Missing feature values
(e.g. first few games of a season before rolling windows fill) are filled
with 0, which is near-neutral for differenced/normalised features.

Usage:
    from src.models.game_outcome import GameOutcomeModel, walk_forward_train, prepare_features

    # With a pre-built feature DataFrame (output of build_game_features):
    model, metrics = walk_forward_train(feature_df)
    print(metrics)

    # Inference on new games:
    probs = model.predict_proba(X_new)[:, 1]   # P(home win)
"""

import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from src.models.calibration import calibrate_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

_ELO_COLS = ["HOME_ELO", "AWAY_ELO", "ELO_DIFF"]

_FORM_COLS = [
    "HOME_STREAK", "AWAY_STREAK",
    "HOME_WINS_L10", "AWAY_WINS_L10",
]

_SCHEDULE_COLS = ["HOME_B2B", "AWAY_B2B", "HOME_REST", "AWAY_REST"]

_H2H_COLS = ["HOME_REC", "AWAY_REC"]

_RATING_COLS = [
    f"{side}_{stat}_L{w}"
    for side in ["HOME", "AWAY"]
    for stat in ["OFF_RATING", "DEF_RATING", "NET_RATING"]
    for w in [5, 10, 20]
]

_PACE_COLS = [f"PROJ_PACE_L{w}" for w in [5, 10, 20]]

_FOUR_FACTOR_COLS = [
    f"{side}_{ff}_L{w}"
    for side in ["HOME", "AWAY"]
    for ff in ["EFG_PCT", "TOV_PCT", "ORB_PCT", "FTR"]
    for w in [5, 10, 20]
]

ALL_FEATURE_COLS = (
    _ELO_COLS + _FORM_COLS + _SCHEDULE_COLS + _H2H_COLS
    + _RATING_COLS + _PACE_COLS + _FOUR_FACTOR_COLS
)

# Walk-forward season split (per CLAUDE.md)
_TRAIN_SEASONS = list(range(2015, 2023))   # 2015–2022
_VAL_SEASON    = 2023
_TEST_SEASON   = 2024

_DEFAULT_XGB_PARAMS: dict = {
    "n_estimators":    400,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "eval_metric":     "logloss",
    "random_state":    42,
    "n_jobs":          -1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GameOutcomeModel:
    """XGBoost-based game outcome classifier.

    Wraps xgb.XGBClassifier and tracks which feature columns are active so
    that predict_proba() always selects the right columns even when called
    with a DataFrame that has extra columns.
    """

    def __init__(self, params: dict | None = None):
        self.params = {**_DEFAULT_XGB_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "GameOutcomeModel":
        """Fit XGBoost on training data.

        Feature columns are determined at fit time as the intersection of
        ALL_FEATURE_COLS and X_train.columns (graceful if some features
        are missing from the DataFrame).

        Args:
            X_train: Training feature matrix.
            y_train: Binary labels (1 = home win).
            X_val:   Optional validation set for tracking learning curves.
            y_val:   Validation labels.

        Returns:
            self (for chaining).
        """
        self.feature_cols = [c for c in ALL_FEATURE_COLS if c in X_train.columns]
        if not self.feature_cols:
            raise ValueError(
                "No recognised feature columns found in X_train. "
                f"Expected at least one of: {ALL_FEATURE_COLS[:5]}..."
            )

        log.info(
            "Training XGBoost on %d features, %d samples.",
            len(self.feature_cols), len(X_train),
        )

        X_fit = X_train[self.feature_cols].fillna(0.0)
        self.model = xgb.XGBClassifier(**self.params)

        if X_val is not None and y_val is not None:
            X_vf = X_val[self.feature_cols].fillna(0.0)
            self.model.fit(X_fit, y_train, eval_set=[(X_vf, y_val)], verbose=False)
        else:
            self.model.fit(X_fit, y_train)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability array of shape (n, 2): [P(away win), P(home win)]."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_f = X[self.feature_cols].fillna(0.0)
        return self.model.predict_proba(X_f)

    def feature_importance(self) -> pd.Series:
        """Feature importances sorted descending by gain."""
        # After calibration self.model is a CalibratedModel wrapper — unwrap it.
        underlying = getattr(self.model, "base_model", self.model)
        return (
            pd.Series(underlying.feature_importances_, index=self.feature_cols)
            .sort_values(ascending=False)
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GameOutcomeModel":
        model = joblib.load(path)
        log.info("Model loaded from %s", path)
        return model


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (ELO_DIFF, LABEL) to a game features DataFrame.

    Computes:
        ELO_DIFF = HOME_ELO - AWAY_ELO   (strongest single predictor)
        LABEL    = 1 if home wins, 0 if away wins

    Drops rows where HOME_PTS == AWAY_PTS (should not occur in NBA but
    guards against bad data).

    Args:
        df: Output of build_game_features() that still contains HOME_PTS
            and AWAY_PTS (outcomes needed for supervised training).

    Returns:
        Copy of df with ELO_DIFF and LABEL columns added.
    """
    df = df.copy()

    if "HOME_ELO" in df.columns and "AWAY_ELO" in df.columns:
        df["ELO_DIFF"] = df["HOME_ELO"] - df["AWAY_ELO"]

    if "HOME_PTS" in df.columns and "AWAY_PTS" in df.columns:
        df = df[df["HOME_PTS"] != df["AWAY_PTS"]].copy()
        df["LABEL"] = (df["HOME_PTS"] > df["AWAY_PTS"]).astype(int)

    return df


def walk_forward_train(
    feature_df: pd.DataFrame,
    train_seasons: list[int] = _TRAIN_SEASONS,
    val_season: int = _VAL_SEASON,
    test_season: int = _TEST_SEASON,
    calibrate: bool = True,
    params: dict | None = None,
) -> tuple["GameOutcomeModel", dict]:
    """Train, optionally calibrate, and evaluate the game outcome model.

    Walk-forward split (never uses future data):
        Train:    seasons in train_seasons   (default 2015–2022)
        Validate: val_season                 (default 2023) — calibration
        Test:     test_season                (default 2024) — final evaluation

    Args:
        feature_df:    Output of build_game_features() with outcome columns.
        train_seasons: List of season ints for training.
        val_season:    Season for calibration / hyperparameter validation.
        test_season:   Held-out test season (report metrics only, never tune).
        calibrate:     Apply isotonic calibration on val set if True.
        params:        XGBoost parameter overrides.

    Returns:
        (model, metrics) — trained model and dict with accuracy/brier/logloss
        for both val and test splits.
    """
    df = prepare_features(feature_df)

    if "LABEL" not in df.columns:
        raise ValueError(
            "feature_df must contain HOME_PTS and AWAY_PTS columns "
            "(needed to compute LABEL for supervised training)."
        )

    train = df[df["SEASON"].isin(train_seasons)]
    val   = df[df["SEASON"] == val_season]
    test  = df[df["SEASON"] == test_season]

    if train.empty:
        raise ValueError(f"No training data for seasons {train_seasons}.")
    if val.empty:
        raise ValueError(f"No validation data for season {val_season}.")
    if test.empty:
        raise ValueError(f"No test data for season {test_season}.")

    log.info(
        "Walk-forward split — train: %d, val: %d, test: %d games",
        len(train), len(val), len(test),
    )

    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X_train, y_train = train[feat_cols], train["LABEL"]
    X_val,   y_val   = val[feat_cols],   val["LABEL"]
    X_test,  y_test  = test[feat_cols],  test["LABEL"]

    model = GameOutcomeModel(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    if calibrate and len(val) >= 50:
        model.model = calibrate_model(
            model.model, X_val.fillna(0.0), y_val, method="isotonic"
        )
        log.info("Applied isotonic calibration on %d val samples.", len(val))

    metrics: dict = {
        "n_features": len(feat_cols),
        "n_train":    len(train),
        "n_val":      len(val),
        "n_test":     len(test),
        **_evaluate_split("val",  model, X_val,  y_val),
        **_evaluate_split("test", model, X_test, y_test),
    }

    log.info(
        "Test — accuracy=%.3f  brier=%.4f  logloss=%.4f",
        metrics["test_accuracy"],
        metrics["test_brier"],
        metrics["test_logloss"],
    )
    return model, metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_split(
    prefix: str,
    model: "GameOutcomeModel",
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Compute accuracy, Brier score, and log loss for one data split."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_accuracy": float((preds == y.values).mean()),
        f"{prefix}_brier":    float(brier_score_loss(y, proba)),
        f"{prefix}_logloss":  float(log_loss(y, proba)),
    }

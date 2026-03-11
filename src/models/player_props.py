"""Player prop prediction models.

Per-stat LightGBM regression models that predict the expected value of a
player's stat for an upcoming game. Predictions are compared to lines
(Underdog or proxied) to identify over/under opportunities.

Stats modeled
-------------
Single:  PTS, REB, AST, FG3M (3-pointers), STL, BLK, TOV
Combo:   PRA (PTS+REB+AST), PR (PTS+REB), PA (PTS+AST), RA (REB+AST)

Three LightGBM models per stat (quantile regression):
    median — predicted value        (α = 0.50)
    low    — 10th percentile bound  (α = 0.10)
    high   — 90th percentile bound  (α = 0.90)

The 80% prediction interval [low, high] gives direct confidence on
over/under: if the line falls below `low`, bet OVER with high confidence.

Usage
-----
    from src.models.player_props import train_all_props, PlayerPropModel

    # Train all stat models
    results = train_all_props(player_feature_df)
    for stat, (model, metrics) in results.items():
        print(stat, metrics)

    # Predict for new games
    preds = model.predict(X_new)
    # preds: {"median": ndarray, "low": ndarray, "high": ndarray}

    # Backtest
    from src.models.player_props import run_prop_backtest
    result = run_prop_backtest(player_feature_df, model, "PTS")
    print(result.summary())
"""

import logging
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stat category definitions
# ---------------------------------------------------------------------------

PROP_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]

COMBO_STATS: dict[str, list[str]] = {
    "PRA": ["PTS", "REB", "AST"],
    "PR":  ["PTS", "REB"],
    "PA":  ["PTS", "AST"],
    "RA":  ["REB", "AST"],
}

ALL_STATS = PROP_STATS + list(COMBO_STATS.keys())

# Walk-forward split (mirrors game outcome model)
_TRAIN_SEASONS = list(range(2015, 2023))
_VAL_SEASON    = 2023
_TEST_SEASON   = 2024

_WINDOWS = [5, 10, 20]

_DEFAULT_LGB_PARAMS: dict = {
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 20,
    "reg_alpha":         0.1,
    "n_jobs":            -1,
    "verbose":           -1,
}

# Quantile levels for prediction intervals
_QUANTILES = {"median": 0.50, "low": 0.10, "high": 0.90}


# ---------------------------------------------------------------------------
# Feature column logic
# ---------------------------------------------------------------------------

def _stat_components(stat: str) -> list[str]:
    """Return the base stats for a given prop (combo or single)."""
    return COMBO_STATS.get(stat, [stat])


def _get_feature_cols(stat: str, available_cols: list[str]) -> list[str]:
    """Build the feature column list for a given stat.

    Includes rolling averages for all component stats, minutes/usage,
    home/away splits for the primary stat, vs-opponent history, and
    contextual signals (B2B, pace, teammate absence).
    """
    components = _stat_components(stat)
    primary    = components[0]

    candidates: list[str] = []

    # Rolling averages for every component stat
    for s in components:
        for w in _WINDOWS:
            candidates.append(f"{s}_L{w}")
        candidates.append(f"{s}_SEASON")

    # Minutes and usage (volume context for all stats)
    for w in _WINDOWS:
        candidates += [f"MIN_L{w}", f"USAGE_PROXY_L{w}"]
    candidates += ["MIN_SEASON", "USAGE_PROXY_SEASON"]

    # Home/away splits for the primary component stat
    candidates += [
        f"{primary}_HOME_AVG",
        f"{primary}_AWAY_AVG",
        f"{primary}_HOME_AWAY_DIFF",
    ]

    # vs-specific-opponent history
    candidates += [f"{primary}_VS_OPP_AVG", "VS_OPP_N"]

    # Schedule and contextual signals
    candidates += [
        "IS_HOME",
        "TEAMMATE_OUT_BOOST",
        "TEAMMATE_OUT_FLAG",
        "PROJ_PACE_L10",
    ]

    return [c for c in candidates if c in available_cols]


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------

class PlayerPropModel:
    """Per-stat player prop model using quantile LightGBM regression.

    Three LightGBM models (median, low, high) are trained simultaneously.
    The median gives the point prediction; [low, high] is the 80% interval.
    """

    def __init__(self, stat: str, params: dict | None = None):
        self.stat    = stat
        self.params  = {**_DEFAULT_LGB_PARAMS, **(params or {})}
        self.model_median: lgb.LGBMRegressor | None = None
        self.model_low:    lgb.LGBMRegressor | None = None
        self.model_high:   lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "PlayerPropModel":
        """Train three quantile models on the training set.

        Args:
            X_train: Training features.
            y_train: Target stat values (continuous, e.g. actual points scored).
            X_val:   Optional validation features (for early stopping info only).
            y_val:   Validation labels.

        Returns:
            self (for chaining).
        """
        self.feature_cols = _get_feature_cols(self.stat, list(X_train.columns))
        if not self.feature_cols:
            raise ValueError(
                f"No recognised feature columns for stat '{self.stat}'. "
                "Run build_player_features() before training."
            )

        X_fit = X_train[self.feature_cols].fillna(0.0)
        log.info(
            "Training %s prop model on %d samples, %d features.",
            self.stat, len(X_train), len(self.feature_cols),
        )

        for attr, alpha in [
            ("model_median", 0.50),
            ("model_low",    0.10),
            ("model_high",   0.90),
        ]:
            model_params = {**self.params, "objective": "quantile", "alpha": alpha}
            model = lgb.LGBMRegressor(**model_params)

            if X_val is not None and y_val is not None:
                X_vf = X_val[self.feature_cols].fillna(0.0)
                model.fit(X_fit, y_train, eval_set=[(X_vf, y_val)])
            else:
                model.fit(X_fit, y_train)

            setattr(self, attr, model)

        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return median, low, and high predictions.

        Returns:
            Dict with keys "median", "low", "high" — each an ndarray of shape (n,).
        """
        if self.model_median is None:
            raise RuntimeError(f"PlayerPropModel({self.stat}) has not been fitted.")
        X_f = X[self.feature_cols].fillna(0.0)
        return {
            "median": self.model_median.predict(X_f),
            "low":    self.model_low.predict(X_f),
            "high":   self.model_high.predict(X_f),
        }

    def feature_importance(self) -> pd.Series:
        """Feature importances of the median model, sorted descending."""
        return (
            pd.Series(
                self.model_median.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        log.info("PlayerPropModel(%s) saved to %s", self.stat, path)

    @classmethod
    def load(cls, path: str) -> "PlayerPropModel":
        model = joblib.load(path)
        log.info("PlayerPropModel loaded from %s", path)
        return model


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_player_features(
    df: pd.DataFrame,
    stat: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) for a given stat.

    For combo stats (PRA, PR, PA, RA), the target is computed as the sum
    of component columns in `df`.

    Args:
        df:   Player feature DataFrame from build_player_features().
        stat: Stat to predict (e.g. "PTS", "PRA").

    Returns:
        (X, y) — feature DataFrame and target Series.

    Raises:
        ValueError: If the stat or its components are not in df.
    """
    df = df.copy()

    if stat in COMBO_STATS:
        components = COMBO_STATS[stat]
        missing = [c for c in components if c not in df.columns]
        if missing:
            raise ValueError(
                f"Combo stat '{stat}' requires columns {components}; "
                f"missing: {missing}"
            )
        df[stat] = df[components].sum(axis=1)
    elif stat not in df.columns:
        raise ValueError(
            f"Stat column '{stat}' not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    df = df.dropna(subset=[stat])
    feat_cols = _get_feature_cols(stat, list(df.columns))
    return df[feat_cols], df[stat].astype(float)


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

def walk_forward_train_prop(
    player_df: pd.DataFrame,
    stat: str,
    train_seasons: list[int] = _TRAIN_SEASONS,
    val_season: int = _VAL_SEASON,
    test_season: int = _TEST_SEASON,
    params: dict | None = None,
) -> tuple[PlayerPropModel, dict]:
    """Train and evaluate a player prop model for one stat.

    Walk-forward split:
        Train:    seasons in train_seasons  (default 2015–2022)
        Validate: val_season                (default 2023)
        Test:     test_season               (default 2024)

    Args:
        player_df:     Output of build_player_features() with actual stat cols.
        stat:          Stat to model (e.g. "PTS", "PRA").
        train_seasons: Seasons for training.
        val_season:    Season for validation.
        test_season:   Held-out test season.
        params:        LightGBM parameter overrides.

    Returns:
        (model, metrics) — trained PlayerPropModel and evaluation dict.
    """
    train = player_df[player_df["SEASON"].isin(train_seasons)]
    val   = player_df[player_df["SEASON"] == val_season]
    test  = player_df[player_df["SEASON"] == test_season]

    if train.empty:
        raise ValueError(f"No training data for seasons {train_seasons}.")
    if val.empty:
        raise ValueError(f"No validation data for season {val_season}.")
    if test.empty:
        raise ValueError(f"No test data for season {test_season}.")

    X_train, y_train = prepare_player_features(train, stat)
    X_val,   y_val   = prepare_player_features(val,   stat)
    X_test,  y_test  = prepare_player_features(test,  stat)

    model = PlayerPropModel(stat=stat, params=params)
    model.fit(X_train, y_train, X_val, y_val)

    metrics = _evaluate_prop(model, X_test, y_test)
    metrics.update({
        "stat":    stat,
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
    })

    log.info(
        "%s — MAE=%.3f  RMSE=%.3f  dir_acc=%.3f  coverage_80=%.3f",
        stat,
        metrics["mae"],
        metrics["rmse"],
        metrics["direction_accuracy"],
        metrics["coverage_80"],
    )
    return model, metrics


def train_all_props(
    player_df: pd.DataFrame,
    stats: list[str] = ALL_STATS,
    train_seasons: list[int] = _TRAIN_SEASONS,
    val_season: int = _VAL_SEASON,
    test_season: int = _TEST_SEASON,
    params: dict | None = None,
) -> dict[str, tuple["PlayerPropModel", dict]]:
    """Train prop models for all stat categories.

    Args:
        player_df:     Output of build_player_features().
        stats:         Stats to model (default: all singles + combos).
        train_seasons: Training seasons.
        val_season:    Validation season.
        test_season:   Test season.
        params:        LightGBM parameter overrides.

    Returns:
        Dict mapping stat name → (model, metrics).
    """
    results: dict[str, tuple[PlayerPropModel, dict]] = {}

    for stat in stats:
        log.info("=" * 45)
        log.info("Training %s prop model ...", stat)
        try:
            model, metrics = walk_forward_train_prop(
                player_df, stat,
                train_seasons=train_seasons,
                val_season=val_season,
                test_season=test_season,
                params=params,
            )
            results[stat] = (model, metrics)
        except Exception as exc:
            log.warning("  %s failed: %s — skipping.", stat, exc)

    return results


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class PropBacktestResult:
    """Per-stat backtest metrics and per-prediction details."""

    stat:            str
    n_games:         int
    n_bets:          int
    direction_accuracy: float   # % of bets where we correctly called OVER/UNDER
    mae:             float      # mean absolute error on all test games
    rmse:            float      # root mean squared error on all test games
    coverage_80:     float      # % of actuals within [low, high] interval
    picks_df:        pd.DataFrame
    edge_threshold:  float

    def summary(self) -> str:
        pct = f"{self.n_bets / self.n_games:.1%}" if self.n_games > 0 else "N/A"
        lines = [
            f"--- {self.stat} Prop Backtest (edge >= {self.edge_threshold:.1f} units) ---",
            f"Test games      : {self.n_games}",
            f"Bets placed     : {self.n_bets}  ({pct} of games)",
            f"Direction acc.  : {self.direction_accuracy:.1%}" if not np.isnan(self.direction_accuracy) else "Direction acc.  : N/A",
            f"MAE (all games) : {self.mae:.3f}",
            f"RMSE (all games): {self.rmse:.3f}",
            f"80% CI coverage : {self.coverage_80:.1%}",
        ]
        return "\n".join(lines)


def run_prop_backtest(
    player_df: pd.DataFrame,
    model: "PlayerPropModel",
    stat: str,
    test_seasons: list[int] | int = _TEST_SEASON,
    edge_threshold: float = 0.5,
) -> PropBacktestResult:
    """Backtest a player prop model on held-out test seasons.

    Since historical Underdog lines are not yet available (Phase 4), the
    "line" is approximated by the player's rolling L10 average
    (``{primary_stat}_L10``). This is a reasonable proxy — Underdog lines
    are typically set near a player's recent rolling average.

    Args:
        player_df:      Output of build_player_features() with actual stats.
        model:          Trained PlayerPropModel.
        stat:           Stat to backtest.
        test_seasons:   Season(s) to evaluate.
        edge_threshold: Minimum |prediction - line| in stat units to place
                        a bet. Default 0.5 (e.g., predict 20.5 vs line 20.0).

    Returns:
        PropBacktestResult with per-pick DataFrame and aggregate metrics.
    """
    if isinstance(test_seasons, int):
        test_seasons = [test_seasons]

    test = player_df[player_df["SEASON"].isin(test_seasons)].copy()
    if test.empty:
        raise ValueError(f"No data for test seasons {test_seasons}.")

    X_test, y_test = prepare_player_features(test, stat)
    preds  = model.predict(X_test)
    median = preds["median"]
    low    = preds["low"]
    high   = preds["high"]

    # Proxy line: rolling L10 average (first available window)
    primary = _stat_components(stat)[0]
    line_col = next(
        (f"{primary}_L{w}" for w in _WINDOWS if f"{primary}_L{w}" in X_test.columns),
        None,
    )
    line = X_test[line_col].fillna(y_test.mean()).values if line_col else np.full(len(y_test), y_test.mean())

    edge      = median - line          # positive → predict OVER
    bet_mask  = np.abs(edge) >= edge_threshold
    bet_over  = edge >= 0              # True → bet OVER
    actual    = y_test.values

    wins = np.where(
        bet_mask,
        (bet_over == (actual > line)).astype(float),
        np.nan,
    )

    # Align with X_test (rows may have been dropped in prepare_player_features)
    picks = test[["PLAYER_NAME", "TEAM", "OPP", "DATE", "SEASON"]].copy()
    picks = picks.loc[X_test.index].reset_index(drop=True)
    picks["ACTUAL"]       = actual
    picks["LINE"]         = line
    picks["PREDICTION"]   = median
    picks["EDGE"]         = edge
    picks["BET"]          = bet_mask.astype(int)
    picks["BET_OVER"]     = bet_over.astype(int)
    picks["WIN"]          = wins
    picks["LOW"]          = low
    picks["HIGH"]         = high

    bets          = picks[picks["BET"] == 1]
    n_bets        = len(bets)
    dir_acc       = float(bets["WIN"].mean()) if n_bets > 0 else float("nan")
    mae           = float(np.abs(median - actual).mean())
    rmse          = float(np.sqrt(((median - actual) ** 2).mean()))
    coverage_80   = float(((actual >= low) & (actual <= high)).mean())

    log.info(
        "%s backtest: %d games, %d bets, dir_acc=%.3f, MAE=%.3f",
        stat, len(test), n_bets, dir_acc if not np.isnan(dir_acc) else 0, mae,
    )

    return PropBacktestResult(
        stat=stat,
        n_games=len(test),
        n_bets=n_bets,
        direction_accuracy=dir_acc,
        mae=mae,
        rmse=rmse,
        coverage_80=coverage_80,
        picks_df=picks,
        edge_threshold=edge_threshold,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_prop(
    model: "PlayerPropModel",
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute MAE, RMSE, direction accuracy, and 80% CI coverage."""
    preds  = model.predict(X_test)
    median = preds["median"]
    low    = preds["low"]
    high   = preds["high"]
    actual = y_test.values

    mae  = float(np.abs(median - actual).mean())
    rmse = float(np.sqrt(((median - actual) ** 2).mean()))

    # Direction accuracy: did our prediction land on the same side of the
    # season mean as the actual value?
    season_mean = float(actual.mean())
    dir_acc = float(((median >= season_mean) == (actual >= season_mean)).mean())

    # 80% prediction interval coverage
    coverage = float(((actual >= low) & (actual <= high)).mean())

    return {
        "mae":                mae,
        "rmse":               rmse,
        "direction_accuracy": dir_acc,
        "coverage_80":        coverage,
    }

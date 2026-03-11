#!/usr/bin/env python
"""Train the game outcome ensemble and run the backtest.

Run this after build_historical_features.py has produced
data/processed/game_features.parquet.

Steps:
    1. Load the feature parquet
    2. Train component models (XGBoost, EloLogit, FourFactorLGB)
    3. Tune ensemble weights on 2023 validation set
    4. Run backtest on 2024 test set
    5. Print results and save model artifacts

Usage:
    python scripts/train_game_outcome.py
    python scripts/train_game_outcome.py --features data/processed/game_features.parquet
    python scripts/train_game_outcome.py --edge-threshold 0.06
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.game_outcome import (
    ALL_FEATURE_COLS,
    GameOutcomeModel,
    _TRAIN_SEASONS,
    _VAL_SEASON,
    _TEST_SEASON,
    prepare_features,
    walk_forward_train,
)
from src.models.ensemble import (
    EloLogitModel,
    EnsembleModel,
    ELO_FEATURE_COLS,
    FF_PACE_FEATURE_COLS,
    FourFactorLGBModel,
)
from src.models.backtest import run_game_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_FEATURES = Path("data/processed/game_features.parquet")
_MODEL_DIR        = Path("data/models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train game outcome model + run backtest.")
    parser.add_argument(
        "--features", type=Path, default=_DEFAULT_FEATURES,
        help="Path to game_features.parquet",
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=0.04,
        help="Minimum edge to place a simulated bet (default: 0.04 = 4%%)",
    )
    parser.add_argument(
        "--no-ensemble", action="store_true",
        help="Skip ensemble training, use XGBoost only",
    )
    args = parser.parse_args()

    if not args.features.exists():
        log.error(
            "Feature file not found: %s\n"
            "Run 'python scripts/build_historical_features.py' first.",
            args.features,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load features
    # -----------------------------------------------------------------------
    log.info("Loading features from %s ...", args.features)
    feature_df = pd.read_parquet(args.features)
    log.info("Loaded %d rows, %d columns", len(feature_df), len(feature_df.columns))

    df = prepare_features(feature_df)
    train = df[df["SEASON"].isin(_TRAIN_SEASONS)]
    val   = df[df["SEASON"] == _VAL_SEASON]
    test  = df[df["SEASON"] == _TEST_SEASON]

    log.info(
        "Split — train: %d games (%s), val: %d (%d), test: %d (%d)",
        len(train), _TRAIN_SEASONS,
        len(val), _VAL_SEASON,
        len(test), _TEST_SEASON,
    )

    feat_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    log.info("Using %d features: %s ...", len(feat_cols), feat_cols[:6])

    X_train, y_train = train[feat_cols], train["LABEL"]
    X_val,   y_val   = val[feat_cols],   val["LABEL"]

    # -----------------------------------------------------------------------
    # 2. Train XGBoost (primary model) — walk-forward with calibration
    # -----------------------------------------------------------------------
    log.info("=" * 55)
    log.info("Training XGBoost game outcome model ...")
    xgb_model, xgb_metrics = walk_forward_train(
        feature_df,
        train_seasons=_TRAIN_SEASONS,
        val_season=_VAL_SEASON,
        test_season=_TEST_SEASON,
        calibrate=True,
    )

    log.info("XGBoost results:")
    log.info("  Val  — accuracy=%.3f  brier=%.4f  logloss=%.4f",
             xgb_metrics["val_accuracy"], xgb_metrics["val_brier"], xgb_metrics["val_logloss"])
    log.info("  Test — accuracy=%.3f  brier=%.4f  logloss=%.4f",
             xgb_metrics["test_accuracy"], xgb_metrics["test_brier"], xgb_metrics["test_logloss"])

    top_features = xgb_model.feature_importance().head(10)
    log.info("Top 10 features by importance:\n%s", top_features.to_string())

    # -----------------------------------------------------------------------
    # 3. Train ensemble components + tune weights
    # -----------------------------------------------------------------------
    final_model = xgb_model

    if not args.no_ensemble:
        log.info("=" * 55)
        log.info("Training EloLogit (Model B) ...")
        elo_model = EloLogitModel().fit(X_train, y_train)

        log.info("Training FourFactorLGB (Model C) ...")
        ff_model = FourFactorLGBModel().fit(X_train, y_train, X_val, y_val)

        log.info("Building ensemble and tuning weights on val set ...")
        ensemble = EnsembleModel(
            models=[xgb_model, elo_model, ff_model],
            feature_col_sets=[ALL_FEATURE_COLS, ELO_FEATURE_COLS, FF_PACE_FEATURE_COLS],
        )
        ensemble.tune_weights(X_val, y_val)
        log.info(
            "Ensemble weights — XGB: %.3f  EloLogit: %.3f  FourFactor: %.3f",
            *ensemble.weights,
        )
        final_model = ensemble

    # -----------------------------------------------------------------------
    # 4. Backtest on 2024 test set
    # -----------------------------------------------------------------------
    log.info("=" * 55)
    log.info("Running backtest on %d season (edge >= %.0f%%) ...",
             _TEST_SEASON, args.edge_threshold * 100)

    backtest_model = xgb_model   # backtest uses the standalone XGBoost (has predict_proba)
    result = run_game_backtest(
        feature_df,
        backtest_model,
        test_seasons=_TEST_SEASON,
        edge_threshold=args.edge_threshold,
    )

    log.info("=" * 55)
    print("\n" + result.summary())
    print()

    # Calibration table
    cal = result.calibration
    print("Calibration (predicted vs actual win rate):")
    print(cal.to_string(index=False, float_format="{:.3f}".format))
    print()

    # -----------------------------------------------------------------------
    # 5. Save model artifacts
    # -----------------------------------------------------------------------
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    xgb_path = _MODEL_DIR / "game_outcome_xgb.joblib"
    xgb_model.save(str(xgb_path))
    log.info("XGBoost model saved to %s", xgb_path)

    picks_path = _MODEL_DIR / "backtest_picks_2024.csv"
    result.picks_df.to_csv(picks_path, index=False)
    log.info("Backtest picks saved to %s", picks_path)

    log.info("Done.")


if __name__ == "__main__":
    main()

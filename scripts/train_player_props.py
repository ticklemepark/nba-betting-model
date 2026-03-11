#!/usr/bin/env python
"""Train player prop models and run backtests for all stat categories.

Run this after build_player_features.py has produced
data/processed/player_features.parquet.

Steps:
    1. Load the player feature parquet
    2. Train one LightGBM quantile regressor per stat (median + 10th + 90th)
    3. Run backtest on 2024 test set for each stat
    4. Print summary table of all results
    5. Save model artifacts to data/models/

Usage:
    python scripts/train_player_props.py
    python scripts/train_player_props.py --stats PTS REB AST PRA
    python scripts/train_player_props.py --edge-threshold 1.0
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.player_props import (
    ALL_STATS,
    PlayerPropModel,
    PropBacktestResult,
    run_prop_backtest,
    train_all_props,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_FEATURES = Path("data/processed/player_features.parquet")
_MODEL_DIR        = Path("data/models")
_TEST_SEASON      = 2024


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train player prop models and run backtest."
    )
    parser.add_argument(
        "--features", type=Path, default=_DEFAULT_FEATURES,
        help="Path to player_features.parquet",
    )
    parser.add_argument(
        "--stats", nargs="+", default=ALL_STATS,
        help=f"Stats to train (default: {ALL_STATS})",
    )
    parser.add_argument(
        "--edge-threshold", type=float, default=0.5,
        help="Minimum |prediction - line| in stat units to place a bet (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.features.exists():
        log.error(
            "Feature file not found: %s\n"
            "Run 'python scripts/build_player_features.py' first.",
            args.features,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load feature matrix
    # -----------------------------------------------------------------------
    log.info("Loading player features from %s ...", args.features)
    player_df = pd.read_parquet(args.features)
    log.info(
        "Loaded %d player-game rows, %d columns, %d unique players.",
        len(player_df),
        len(player_df.columns),
        player_df["PLAYER_ID"].nunique() if "PLAYER_ID" in player_df.columns else -1,
    )

    # -----------------------------------------------------------------------
    # 2. Train all prop models
    # -----------------------------------------------------------------------
    log.info("Training prop models for: %s", args.stats)
    results = train_all_props(player_df, stats=args.stats)

    if not results:
        log.error("No models trained successfully.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Backtest each model and collect summary rows
    # -----------------------------------------------------------------------
    log.info("=" * 55)
    log.info("Running backtests on %d season ...", _TEST_SEASON)

    summary_rows = []
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for stat, (model, train_metrics) in results.items():
        log.info("Backtesting %s ...", stat)
        try:
            bt = run_prop_backtest(
                player_df,
                model,
                stat,
                test_seasons=_TEST_SEASON,
                edge_threshold=args.edge_threshold,
            )
            summary_rows.append({
                "Stat":       stat,
                "N_test":     bt.n_games,
                "N_bets":     bt.n_bets,
                "Dir_acc":    f"{bt.direction_accuracy:.1%}" if bt.n_bets > 0 else "N/A",
                "MAE":        f"{bt.mae:.3f}",
                "RMSE":       f"{bt.rmse:.3f}",
                "Coverage80": f"{bt.coverage_80:.1%}",
            })

            # Save per-stat picks
            picks_path = _MODEL_DIR / f"prop_picks_{stat}_2024.csv"
            bt.picks_df.to_csv(picks_path, index=False)

        except Exception as exc:
            log.warning("  %s backtest failed: %s", stat, exc)
            summary_rows.append({
                "Stat": stat, "N_test": "—", "N_bets": "—",
                "Dir_acc": "ERROR", "MAE": "—", "RMSE": "—", "Coverage80": "—",
            })

        # Save model artifact
        model_path = _MODEL_DIR / f"player_prop_{stat.lower()}.joblib"
        model.save(str(model_path))

    # -----------------------------------------------------------------------
    # 4. Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("PLAYER PROP MODEL RESULTS — 2024 TEST SEASON")
    print("=" * 55)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()
    print(f"Edge threshold: {args.edge_threshold} stat units")
    print(f"Line proxy: player's rolling L10 average (real lines in Phase 4)")
    print(f"Models saved to: {_MODEL_DIR}/")

    log.info("Done.")


if __name__ == "__main__":
    main()

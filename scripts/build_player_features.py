#!/usr/bin/env python
"""Build the historical player feature matrix for prop model training.

Fetches player and team game logs for all seasons from nba_api, runs the
full player feature pipeline, and saves the result as a Parquet file.

Runtime: ~5–15 minutes (20 API calls: 10 player + 10 team, one per season).

Usage:
    # All seasons 2015-2024 (default)
    python scripts/build_player_features.py

    # Specific seasons only
    python scripts/build_player_features.py --seasons 2023 2024

    # Custom output path
    python scripts/build_player_features.py --out data/processed/player_features.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import get_cursor
from src.data.nba_api_client import fetch_player_game_logs, fetch_team_game_logs
from src.features.pipeline import build_player_features
from src.models.player_props import COMBO_STATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_SEASONS = list(range(2015, 2025))
_DEFAULT_OUT     = Path("data/processed/player_features.parquet")
_DEFAULT_WINDOWS = [5, 10, 20]
_PLAYER_STATS    = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
_API_SLEEP       = 1.0


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def load_games_from_db(seasons: list[int]) -> pd.DataFrame:
    """Load game schedule + results from PostgreSQL."""
    placeholders = ", ".join(["%s"] * len(seasons))
    with get_cursor() as cur:
        cur.execute(
            f"SELECT home, away, home_pts, away_pts, date::text, season "
            f"FROM games WHERE season IN ({placeholders}) ORDER BY date",
            seasons,
        )
        rows = cur.fetchall()
        cols = [d.name.upper() for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def fetch_logs_for_seasons(
    seasons: list[int],
    fetch_fn,
    label: str,
) -> pd.DataFrame:
    """Fetch game logs for each season, concatenate, and return."""
    parts = []
    for season in seasons:
        log.info("Fetching %s logs for season %d ...", label, season)
        try:
            df = fetch_fn(season)
            parts.append(df)
            log.info("  %d rows", len(df))
        except Exception as exc:
            log.warning("  Season %d failed: %s — skipping.", season, exc)
        time.sleep(_API_SLEEP)

    if not parts:
        raise RuntimeError(f"No {label} log data fetched.")
    combined = pd.concat(parts, ignore_index=True)
    log.info("Fetched %d total %s rows.", len(combined), label)
    return combined


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------

def add_combo_stat_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add combo stat columns (PRA, PR, PA, RA) as model targets."""
    for combo, components in COMBO_STATS.items():
        available = [c for c in components if c in df.columns]
        if len(available) == len(components):
            df[combo] = df[available].sum(axis=1)
        else:
            log.warning(
                "Combo stat %s skipped — missing columns: %s",
                combo, [c for c in components if c not in df.columns],
            )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical player feature matrix."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=_DEFAULT_SEASONS,
        help="Seasons to include (default: 2015-2024)",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="Output Parquet path",
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=_DEFAULT_WINDOWS,
        help="Rolling window sizes (default: 5 10 20)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Building player feature matrix")
    log.info("  Seasons : %s", args.seasons)
    log.info("  Windows : %s", args.windows)
    log.info("  Output  : %s", args.out)
    log.info("=" * 60)

    # 1. Load games from DB (for schedule context)
    games = load_games_from_db(args.seasons)
    if games.empty:
        log.error("No games in database for seasons %s.", args.seasons)
        sys.exit(1)
    log.info("Loaded %d games from database.", len(games))

    # 2. Fetch player game logs
    player_logs = fetch_logs_for_seasons(
        args.seasons, fetch_player_game_logs, "player"
    )

    # 3. Fetch team game logs (needed for pace context)
    team_logs = fetch_logs_for_seasons(
        args.seasons, fetch_team_game_logs, "team"
    )

    # 4. Build player feature matrix
    # For historical backtesting, player_logs serves as both the "games to
    # predict for" and the "historical data". Zero-leakage is guaranteed by
    # the merge_asof direction="backward" in each feature module.
    log.info("Running player feature pipeline ...")
    player_features = build_player_features(
        player_games=player_logs,
        player_logs=player_logs,
        team_logs=team_logs,
        absent_players=None,      # no DNP data for historical
        stats=_PLAYER_STATS,
        windows=args.windows,
    )
    log.info("Player feature matrix shape: %s", player_features.shape)

    # 5. Add combo stat target columns (PRA, PR, PA, RA)
    player_features = add_combo_stat_targets(player_features)

    # 6. Sanity check
    n_players = player_features["PLAYER_ID"].nunique() if "PLAYER_ID" in player_features.columns else "?"
    n_seasons  = player_features["SEASON"].nunique()   if "SEASON"    in player_features.columns else "?"
    log.info(
        "Sanity — %s unique players across %s seasons.",
        n_players, n_seasons,
    )

    # 7. Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    player_features.to_parquet(args.out, index=False)
    log.info("Saved to %s", args.out)
    log.info("Done. Run 'python scripts/train_player_props.py' to train models.")


if __name__ == "__main__":
    main()

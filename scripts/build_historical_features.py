#!/usr/bin/env python
"""Build the historical game feature matrix for model training.

Fetches team game logs for all seasons from nba_api, runs the full
feature pipeline (ELO, streaks, B2B, rest, H2H, rolling ratings,
pace, Four Factors), and saves the resulting feature matrix as a
Parquet file ready for model training.

Runtime: ~2-5 minutes (10 API calls, one per season).

Usage:
    # All seasons 2015-2024 (default)
    python scripts/build_historical_features.py

    # Specific seasons
    python scripts/build_historical_features.py --seasons 2023 2024

    # Custom output path
    python scripts/build_historical_features.py --out data/processed/features.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db import get_cursor
from src.data.nba_api_client import fetch_team_game_logs
from src.features.pipeline import build_game_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_SEASONS  = list(range(2015, 2025))   # 2015–2024
_DEFAULT_OUT      = Path("data/processed/game_features.parquet")
_DEFAULT_WINDOWS  = [5, 10, 20]
_API_SLEEP        = 1.0   # courtesy pause between nba_api calls


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_games_from_db(seasons: list[int]) -> pd.DataFrame:
    """Load game schedule + results from PostgreSQL.

    Returns a DataFrame with columns:
        HOME, AWAY, HOME_PTS, AWAY_PTS, DATE, SEASON
    """
    placeholders = ", ".join(["%s"] * len(seasons))
    query = f"""
        SELECT home, away, home_pts, away_pts, date::text, season
        FROM games
        WHERE season IN ({placeholders})
        ORDER BY date
    """
    with get_cursor() as cur:
        cur.execute(query, seasons)
        rows = cur.fetchall()
        cols = [d.name.upper() for d in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    log.info("Loaded %d games from database (seasons %s).", len(df), seasons)
    return df


def fetch_all_team_logs(seasons: list[int]) -> pd.DataFrame:
    """Fetch team game logs for each season from nba_api (one call per season)."""
    parts = []
    for season in seasons:
        log.info("Fetching team logs for season %d ...", season)
        try:
            df = fetch_team_game_logs(season)
            parts.append(df)
            log.info("  %d team-game rows", len(df))
        except Exception as exc:
            log.warning("  Season %d failed: %s — skipping.", season, exc)
        time.sleep(_API_SLEEP)

    if not parts:
        raise RuntimeError(
            "No team log data could be fetched. Check your internet connection "
            "and that nba_api is installed."
        )

    combined = pd.concat(parts, ignore_index=True)
    log.info(
        "Fetched %d total team-game rows across %d seasons.",
        len(combined), len(parts),
    )
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical game feature matrix."
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
    log.info("Building game feature matrix")
    log.info("  Seasons : %s", args.seasons)
    log.info("  Windows : %s", args.windows)
    log.info("  Output  : %s", args.out)
    log.info("=" * 60)

    # 1. Load games from DB
    games = load_games_from_db(args.seasons)
    if games.empty:
        log.error("No games found in database for the requested seasons.")
        sys.exit(1)

    # 2. Fetch team logs from nba_api
    team_logs = fetch_all_team_logs(args.seasons)

    # 3. Build full feature matrix
    log.info("Running feature pipeline ...")
    features = build_game_features(games, team_logs, windows=args.windows)
    log.info("Feature matrix shape: %s", features.shape)

    # 4. Quick sanity check
    n_complete = features["HOME_PTS"].notna().sum()
    n_with_ratings = (
        features[[c for c in features.columns if "NET_RATING" in c]]
        .notna().any(axis=1).sum()
        if any("NET_RATING" in c for c in features.columns) else 0
    )
    log.info(
        "Sanity — games with pts: %d/%d, games with any rating: %d/%d",
        n_complete, len(features), n_with_ratings, len(features),
    )

    # 5. Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    log.info("Saved to %s", args.out)
    log.info("Done. Run 'python -m src.models.game_outcome' to train the model.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build game feature rows for today's scheduled NBA games.

Unlike build_historical_features.py (which reads from the PostgreSQL games
table), this script reconstructs the current season's game history directly
from nba_api team logs and appends today's scheduled matchups so that
daily_pipeline.py can find today's rows in game_features.parquet.

Workflow:
    1. Load historical game rows (2015-2024) from existing game_features.parquet
       — used as the base for ELO / streak / B2B continuity.
    2. Fetch current-season (2026) team logs from nba_api.
    3. Reconstruct played 2026 games from team logs (home/away from IS_HOME flag).
    4. Fetch today's scheduled games from nba_api ScoreboardV2.
    5. Combine: [historical 2015-2024 games] + [played 2026 games] + [today].
    6. Run build_game_features on the full combined dataset.
    7. Extract today's row(s) and upsert into game_features.parquet.

Usage:
    # Build for today (default)
    python scripts/build_today_game_features.py

    # Build for a specific date
    python scripts/build_today_game_features.py --date 2026-03-10

    # Force-rebuild even if today already has rows
    python scripts/build_today_game_features.py --force
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nba_api_client import fetch_team_game_logs
from src.data.scrapers.injury_report import fetch_injury_report
from src.features.pipeline import build_game_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_GAME_FEATURES_PATH  = Path("data/processed/game_features.parquet")
_INJURY_REPORT_PATH  = Path("data/processed/injury_report_today.csv")
_CURRENT_SEASON      = 2026   # 2025-26 season
_API_SLEEP           = 1.0


# ---------------------------------------------------------------------------
# Step 1: load historical games from existing parquet
# ---------------------------------------------------------------------------

def load_historical_games(parquet_path: Path) -> pd.DataFrame:
    """Load HOME, AWAY, HOME_PTS, AWAY_PTS, DATE, SEASON from existing parquet."""
    if not parquet_path.exists():
        log.warning("Historical parquet not found at %s — starting from scratch.", parquet_path)
        return pd.DataFrame(columns=["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"])

    df = pd.read_parquet(parquet_path)
    needed = ["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.error("game_features.parquet is missing columns: %s", missing)
        log.error("Run build_historical_features.py first.")
        sys.exit(1)

    # Keep only historical seasons (exclude any current-season rows already written)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")
    historical = df[df["SEASON"] < _CURRENT_SEASON][needed].copy()
    log.info("Loaded %d historical game rows (seasons 2015-%d).",
             len(historical), _CURRENT_SEASON - 1)
    return historical


# ---------------------------------------------------------------------------
# Step 2: fetch current season team logs
# ---------------------------------------------------------------------------

def fetch_current_team_logs() -> pd.DataFrame:
    log.info("Fetching %d team game logs from nba_api ...", _CURRENT_SEASON)
    df = fetch_team_game_logs(_CURRENT_SEASON)
    time.sleep(_API_SLEEP)
    log.info("Fetched %d team-game rows for season %d.", len(df), _CURRENT_SEASON)
    return df


# ---------------------------------------------------------------------------
# Step 3: reconstruct played games from team logs
# ---------------------------------------------------------------------------

def reconstruct_games_from_logs(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert team-level logs into one-row-per-game format.

    team_logs has one row per team per game.  IS_HOME=True rows give us
    HOME team info; IS_HOME=False rows give us AWAY team info.
    We join on GAME_ID to get HOME_PTS and AWAY_PTS in the same row.
    """
    if team_logs.empty:
        log.warning("team_logs is empty — no 2026 games reconstructed.")
        return pd.DataFrame(columns=["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"])

    home = (
        team_logs[team_logs["IS_HOME"] == True]
        [["GAME_ID", "TEAM", "OPP", "PTS", "DATE", "SEASON"]]
        .rename(columns={"TEAM": "HOME", "OPP": "AWAY", "PTS": "HOME_PTS"})
    )
    away = (
        team_logs[team_logs["IS_HOME"] == False]
        [["GAME_ID", "PTS"]]
        .rename(columns={"PTS": "AWAY_PTS"})
    )

    games = home.merge(away, on="GAME_ID", how="inner")
    games = games[["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"]]
    games = games.drop_duplicates(subset=["HOME", "AWAY", "DATE"]).reset_index(drop=True)
    log.info("Reconstructed %d played 2026 games.", len(games))
    return games


# ---------------------------------------------------------------------------
# Step 4: fetch today's scheduled games
# ---------------------------------------------------------------------------

def fetch_todays_games(target_date: date) -> pd.DataFrame:
    """Return a DataFrame of today's scheduled matchups with NaN scores.

    Uses ScoreboardV3 (recommended for 2025-26 season).
    gameCode format: '{date}/{awayTricode}{homeTricode}' e.g. '20260310/MEMPHI'
    """
    date_str = target_date.strftime("%Y-%m-%d")
    log.info("Fetching today's schedule for %s ...", date_str)

    try:
        from nba_api.stats.endpoints import scoreboardv3

        board = scoreboardv3.ScoreboardV3(
            game_date=date_str,
            league_id="00",
            timeout=30,
        )
        time.sleep(_API_SLEEP)
        dfs = board.get_data_frames()
        games_header = dfs[1]   # gameId, gameCode, gameStatus, ...

        rows = []
        for _, row in games_header.iterrows():
            game_code = str(row.get("gameCode", ""))
            # Format: "20260310/MEMPHI" -> suffix "MEMPHI" -> away="MEM", home="PHI"
            suffix = game_code.split("/")[-1] if "/" in game_code else ""
            if len(suffix) == 6:
                away_abbr = suffix[:3]
                home_abbr = suffix[3:]
            else:
                log.warning("Unexpected gameCode format: %s — skipping.", game_code)
                continue

            rows.append({
                "HOME":     home_abbr,
                "AWAY":     away_abbr,
                "HOME_PTS": None,
                "AWAY_PTS": None,
                "DATE":     date_str,
                "SEASON":   _CURRENT_SEASON,
            })

        df = pd.DataFrame(rows)
        log.info("Found %d games scheduled for %s: %s",
                 len(df), date_str,
                 [f"{r.AWAY}@{r.HOME}" for r in df.itertuples()])
        return df

    except Exception as exc:
        log.warning("Could not fetch today's schedule: %s", exc)
        return pd.DataFrame(columns=["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"])


# ---------------------------------------------------------------------------
# Step 4b: fetch and save injury report for today's teams
# ---------------------------------------------------------------------------

def fetch_and_save_injuries(today_games: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Fetch ESPN injury report and filter to teams playing today.

    Saves filtered injuries to a CSV so daily_pipeline.py can load them
    without re-fetching.

    Args:
        today_games: DataFrame with HOME and AWAY columns (today's matchups).
        out_path:    Path to write injury_report_today.csv.

    Returns:
        DataFrame of injuries for today's teams only.
    """
    log.info("Fetching injury report from ESPN JSON API ...")
    injuries = fetch_injury_report()

    if injuries.empty:
        log.warning("No injury data returned — injury report will be empty.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["PLAYER_NAME", "TEAM", "STATUS", "REASON", "FETCHED_AT"]).to_csv(out_path, index=False)
        return injuries

    log.info("Fetched %d total injury entries.", len(injuries))

    # Filter to teams playing today
    if not today_games.empty:
        today_teams = set(today_games["HOME"].tolist()) | set(today_games["AWAY"].tolist())
        injuries_today = injuries[injuries["TEAM"].isin(today_teams)].copy()
        log.info(
            "Filtered to %d injury entries for today's %d teams.",
            len(injuries_today), len(today_teams),
        )
    else:
        injuries_today = injuries.copy()

    # Log OUT and DOUBTFUL players prominently
    critical = injuries_today[injuries_today["STATUS"].isin(["OUT", "DOUBTFUL"])]
    if not critical.empty:
        log.info("OUT / DOUBTFUL players for today's games:")
        for _, row in critical.iterrows():
            log.info("  %-4s  %-25s  %-12s  %s",
                     row["TEAM"], row["PLAYER_NAME"], row["STATUS"], row["REASON"])
    else:
        log.info("No OUT/DOUBTFUL players for today's games.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    injuries_today.to_csv(out_path, index=False)
    log.info("Saved injury report to %s", out_path)
    return injuries_today


# ---------------------------------------------------------------------------
# Step 5-7: combine, build features, upsert
# ---------------------------------------------------------------------------

def build_and_upsert(
    historical: pd.DataFrame,
    games_2026: pd.DataFrame,
    today_games: pd.DataFrame,
    team_logs_2026: pd.DataFrame,
    target_date: date,
    out_path: Path,
) -> pd.DataFrame:
    """Run the feature pipeline and write today's rows to game_features.parquet.

    Strategy: Run the feature pipeline only on games with known scores
    (historical + played 2026 games). Then for today's unplayed games, we
    compute ELO/streak/B2B/etc. by appending placeholder rows and running a
    second pass — but since ELO would crash on None scores, we instead use a
    sentinel approach: substitute NaN scores with 0 to let ELO record
    pre-game values, then discard the post-game ELO update for those rows.

    Simpler approach used here: run the pipeline on [historical + 2026 played],
    then append today's games with a minimal stub feature pass (ELO only,
    using the terminal ELO state from the previous run).
    """
    date_str = target_date.strftime("%Y-%m-%d")

    # Combine historical + played 2026 games (all have known scores)
    played_games = pd.concat([historical, games_2026], ignore_index=True)
    played_games = played_games.sort_values("DATE").reset_index(drop=True)
    log.info("Running feature pipeline on %d played games ...", len(played_games))

    # Run full pipeline on played games — ELO, streaks, B2B, rolling ratings, etc.
    features_played = build_game_features(played_games, team_logs_2026)
    log.info("Played feature matrix shape: %s", features_played.shape)

    if today_games.empty:
        log.warning("No today games to feature — done.")
        return pd.DataFrame()

    # For today's games (no scores), we need pre-game ELO and other pre-game features.
    # Trick: append today's games with dummy scores (0-0), run pipeline, then strip
    # the post-game ELO update. The pre-game ELO (HOME_ELO, AWAY_ELO) will be correct
    # since it uses terminal ELO state from the played-games run.
    today_dummy = today_games.copy()
    today_dummy["HOME_PTS"] = 0
    today_dummy["AWAY_PTS"] = 0

    all_with_dummy = pd.concat([played_games, today_dummy], ignore_index=True)
    all_with_dummy = all_with_dummy.sort_values("DATE").reset_index(drop=True)
    log.info("Running pipeline with dummy scores for today's %d games ...", len(today_games))
    features_all = build_game_features(all_with_dummy, team_logs_2026)

    # Extract today's rows and restore NaN scores
    features_all["DATE"] = pd.to_datetime(features_all["DATE"]).dt.strftime("%Y-%m-%d")
    today_features = features_all[features_all["DATE"] == date_str].copy()
    today_features["HOME_PTS"] = None
    today_features["AWAY_PTS"] = None
    # HOME_AFTER and AWAY_AFTER (post-game ELO) are meaningless for unplayed games
    if "HOME_AFTER" in today_features.columns:
        today_features["HOME_AFTER"] = None
        today_features["AWAY_AFTER"] = None

    log.info("Today's feature rows: %d", len(today_features))

    if today_features.empty:
        log.error(
            "No feature rows for %s after running pipeline. "
            "Check that ScoreboardV3 returned games and the pipeline ran correctly.",
            date_str,
        )
        return today_features

    # Upsert: load existing parquet, remove any stale today rows, append fresh ones
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["DATE"] = pd.to_datetime(existing["DATE"]).dt.strftime("%Y-%m-%d")
        existing_without_today = existing[existing["DATE"] != date_str]
        updated = pd.concat([existing_without_today, today_features], ignore_index=True)
        log.info(
            "Upserted %d today rows into existing parquet (%d -> %d total rows).",
            len(today_features), len(existing), len(updated),
        )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        updated = today_features
        log.info("Creating new parquet with %d rows.", len(updated))

    updated.to_parquet(out_path, index=False)
    log.info("Saved to %s", out_path)
    return today_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build game feature rows for today's scheduled NBA games."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--out", type=Path, default=_GAME_FEATURES_PATH,
        help="Output parquet path"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-build even if today already has rows in the parquet"
    )
    parser.add_argument(
        "--injury-out", type=Path, default=_INJURY_REPORT_PATH,
        help="Path to write today's injury report CSV",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    log.info("=" * 60)
    log.info("build_today_game_features — %s", target_date)
    log.info("=" * 60)

    # Quick check: if today's rows already exist and --force not set, skip
    if not args.force and args.out.exists():
        existing = pd.read_parquet(args.out)
        date_col = next((c for c in existing.columns if c.upper() == "DATE"), None)
        if date_col:
            existing[date_col] = pd.to_datetime(existing[date_col]).dt.strftime("%Y-%m-%d")
            if (existing[date_col] == target_date.strftime("%Y-%m-%d")).any():
                log.info("Today's rows already exist in %s — done (use --force to rebuild).", args.out)
                return

    # 1. Historical games from parquet
    historical = load_historical_games(args.out)

    # 2. Current season team logs
    team_logs_2026 = fetch_current_team_logs()

    # 3. Reconstruct played 2026 games
    games_2026 = reconstruct_games_from_logs(team_logs_2026)

    # 4. Today's scheduled games
    today_games = fetch_todays_games(target_date)

    if today_games.empty:
        log.warning(
            "No games scheduled for %s (or ScoreboardV3 failed). "
            "Check the date or try again later.", target_date,
        )

    # 4b. Injury report for today's teams
    fetch_and_save_injuries(today_games, args.injury_out)

    # 5-7. Build features and upsert
    today_features = build_and_upsert(
        historical, games_2026, today_games, team_logs_2026, target_date, args.out
    )

    if today_features.empty:
        log.error("No features built for today. daily_pipeline.py will still find no game features.")
        sys.exit(1)

    log.info("Done. Run daily_pipeline.py to generate picks.")


if __name__ == "__main__":
    main()

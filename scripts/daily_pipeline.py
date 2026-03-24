#!/usr/bin/env python
"""Daily Underdog Fantasy pick pipeline.

Workflow:
    1. Fetch today's Underdog prop + game lines (or use --dry-run with saved models)
    2. Load today's feature rows (from data/processed/ parquets)
    3. Load trained models from data/models/
    4. Screen prop picks  → list[PropPick]
    5. Screen game picks  → list[GamePick]
    6. Build optimal entries (correlation-aware)
    7. Size each entry (fractional Kelly)
    8. Print daily pick sheet
    9. Log entries to database (unless --dry-run)

Usage:
    # Live run (requires UNDERDOG_TOKEN in .env)
    python scripts/daily_pipeline.py

    # Dry run — no DB writes, no Underdog API calls (uses empty line pools)
    python scripts/daily_pipeline.py --dry-run

    # Specify parameters
    python scripts/daily_pipeline.py \\
        --date 2025-01-15 \\
        --bankroll 1000 \\
        --min-edge 0.05 \\
        --max-entries 10 \\
        --kelly 0.25
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.edge_calculator import screen_game_picks, screen_prop_picks
from src.betting.entry_builder import build_entries, rank_entries
from src.betting.kelly import UNDERDOG_PAYOUTS, size_entry, summarise_sizing
from src.betting.line_divergence import detect_divergences, log_divergences_to_db
from src.betting.tracker import log_entry
from src.data.scrapers.underdog import (
    UnderdogAuthError,
    fetch_game_lines,
    fetch_prop_lines,
    save_lines_to_db,
)
from src.models.game_outcome import ALL_FEATURE_COLS, GameOutcomeModel, prepare_features
from src.models.player_props import ALL_STATS, PlayerPropModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_GAME_FEATURES_PATH   = Path("data/processed/game_features.parquet")
_PLAYER_FEATURES_PATH = Path("data/processed/player_features.parquet")
_INJURY_REPORT_PATH   = Path("data/processed/injury_report_today.csv")
_MODEL_DIR            = Path("data/models")
_DEFAULT_BANKROLL     = 1000.0
_DEFAULT_MIN_EDGE     = 0.04
_DEFAULT_MAX_ENTRIES  = 15

# Priority stat filter — only surface these bet types until explicitly expanded.
# AST restricted to primary ball-handlers (L10 avg > threshold).
# REB restricted to centers/bigs (L10 avg > threshold).
_PRIORITY_STATS         = {"PTS", "PRA", "AST", "FG3M", "REB"}
_AST_PRIMARY_HANDLER_L10 = 4.5   # min AST L10 avg to qualify for AST picks
_REB_CENTER_L10          = 6.5   # min REB L10 avg to qualify for REB picks
_DEFAULT_KELLY        = 0.25


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_game_model() -> GameOutcomeModel | None:
    path = _MODEL_DIR / "game_outcome_xgb.joblib"
    if not path.exists():
        log.warning("Game outcome model not found at %s — skipping game picks.", path)
        return None
    model = GameOutcomeModel.load(str(path))
    log.info("Loaded game outcome model from %s.", path)
    return model


def load_prop_models() -> dict[str, PlayerPropModel]:
    models: dict[str, PlayerPropModel] = {}
    for stat in ALL_STATS:
        path = _MODEL_DIR / f"player_prop_{stat.lower()}.joblib"
        if path.exists():
            models[stat] = PlayerPropModel.load(str(path))
        else:
            log.debug("No prop model for %s at %s — skipping.", stat, path)
    log.info("Loaded %d / %d prop models.", len(models), len(ALL_STATS))
    return models


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_today_game_features(today: date) -> pd.DataFrame:
    """Load game feature rows for today from the parquet cache."""
    if not _GAME_FEATURES_PATH.exists():
        log.warning("Game features not found at %s.", _GAME_FEATURES_PATH)
        return pd.DataFrame()

    df = pd.read_parquet(_GAME_FEATURES_PATH)
    date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        today_df = df[df[date_col] == today].copy()
    else:
        today_df = pd.DataFrame()

    if today_df.empty:
        log.warning(
            "No game features for %s in %s. "
            "Did you run build_historical_features.py today?",
            today, _GAME_FEATURES_PATH,
        )
    return today_df


def load_today_player_features(today: date) -> pd.DataFrame:
    """Load the most-recent player feature row per player from the current season.

    Since today's games haven't been played yet, there are no rows with
    DATE == today.  Instead we use the most recent game row each player has
    (all rolling stats are pre-game, so the last row reflects everything up
    to and including yesterday's games — exactly what we need for tonight).

    Falls back to filtering by today's date first (for backward compatibility
    if a future script pre-populates today's rows).
    """
    if not _PLAYER_FEATURES_PATH.exists():
        log.warning("Player features not found at %s.", _PLAYER_FEATURES_PATH)
        return pd.DataFrame()

    df = pd.read_parquet(_PLAYER_FEATURES_PATH)
    date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
    if date_col is None:
        log.warning("player_features.parquet has no DATE column.")
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Prefer exact today match (future: build_today_player_features.py could populate this)
    today_df = df[df[date_col] == today].copy()
    if not today_df.empty:
        log.info("Loaded %d player feature rows for %s.", len(today_df), today)
        return today_df

    # Fall back: most recent row per player (use current season only if available)
    current_season = today.year if today.month >= 10 else today.year
    current_df = df[df["SEASON"] == current_season] if "SEASON" in df.columns else df
    if current_df.empty:
        current_df = df  # fall back to all seasons

    # Keep most recent row per player
    name_col = next((c for c in df.columns if c.upper() == "PLAYER_NAME"), None)
    if name_col is None:
        log.warning("player_features.parquet has no PLAYER_NAME column.")
        return pd.DataFrame()

    latest = (
        current_df
        .sort_values(date_col)
        .groupby(name_col, sort=False)
        .last()
        .reset_index()
    )
    log.info(
        "No player features for %s — using most recent row per player "
        "(%d players from season %d).",
        today, len(latest), current_season,
    )
    return latest


# ---------------------------------------------------------------------------
# Injury report loading
# ---------------------------------------------------------------------------

def load_injury_report() -> pd.DataFrame:
    """Load today's injury report from the CSV written by build_today_game_features.py.

    Returns an empty DataFrame if the file doesn't exist or can't be read.
    """
    if not _INJURY_REPORT_PATH.exists():
        log.debug("Injury report not found at %s — run build_today_game_features.py first.", _INJURY_REPORT_PATH)
        return pd.DataFrame(columns=["PLAYER_NAME", "TEAM", "STATUS", "REASON"])

    try:
        df = pd.read_csv(_INJURY_REPORT_PATH)
        log.info("Loaded %d injury entries from %s.", len(df), _INJURY_REPORT_PATH)
        return df
    except Exception as exc:
        log.warning("Could not load injury report: %s", exc)
        return pd.DataFrame(columns=["PLAYER_NAME", "TEAM", "STATUS", "REASON"])


_MIN_HISTORICAL_GAMES = 5   # need at least this many "absent" games to trust the delta


def _historical_absent_delta(
    hist: pd.DataFrame,
    absent_name_lower: str,
    teammate_name_lower: str,
    team: str,
    stat_cols: list[str],
    window: int = 5,
) -> dict[str, float] | None:
    """Compute stat delta for a teammate when a specific player is absent.

    Steps:
      1. Partition all historical team games into "absent played" vs "absent sat".
      2. Compute teammate's mean stat in each partition → raw_delta.
      3. Scale by the fraction of the teammate's last `window` games where the
         absent player was actually present.  This prevents double-counting when
         the absent player has already been out for recent games that already
         shaped the teammate's current L5 average.

    Returns None if the sample of absent games is too small (< _MIN_HISTORICAL_GAMES).
    """
    team_hist = hist[hist["TEAM"] == team]
    if team_hist.empty:
        return None

    absent_player_rows = team_hist[team_hist["PLAYER_NAME"].str.lower() == absent_name_lower]
    games_with = set(absent_player_rows["GAME_ID"].dropna())

    all_team_games = set(team_hist["GAME_ID"].dropna())
    games_without  = all_team_games - games_with

    if len(games_without) < _MIN_HISTORICAL_GAMES:
        return None

    teammate_rows = team_hist[team_hist["PLAYER_NAME"].str.lower() == teammate_name_lower]
    if teammate_rows.empty:
        return None

    with_rows    = teammate_rows[teammate_rows["GAME_ID"].isin(games_with)]
    without_rows = teammate_rows[teammate_rows["GAME_ID"].isin(games_without)]

    if len(without_rows) < _MIN_HISTORICAL_GAMES:
        return None

    # Raw per-stat delta: how much better/worse does the teammate perform when absent player sits?
    raw_deltas: dict[str, float] = {}
    for stat in stat_cols:
        if stat not in team_hist.columns:
            continue
        mean_with    = with_rows[stat].mean()    if not with_rows.empty    else None
        mean_without = without_rows[stat].mean() if not without_rows.empty else None
        if mean_with is not None and mean_without is not None:
            raw_deltas[stat] = mean_without - mean_with

    if not raw_deltas:
        return None

    # Scale factor: fraction of teammate's last `window` games where the absent player played.
    # If fraction = 0 → absent player was already out for all recent games → current L5 already
    # reflects absence → apply zero additional delta to avoid double-counting.
    # If fraction = 1 → absent player played every recent game → apply full delta.
    if "DATE" in teammate_rows.columns and "GAME_ID" in teammate_rows.columns:
        recent_games = (
            teammate_rows
            .dropna(subset=["DATE", "GAME_ID"])
            .sort_values("DATE", ascending=False)
            .head(window)["GAME_ID"]
            .tolist()
        )
        n_with_in_window = sum(1 for gid in recent_games if gid in games_with)
        scale = n_with_in_window / max(len(recent_games), 1)
    else:
        scale = 1.0  # no date info → apply full delta (conservative)

    if scale <= 0:
        return {}   # absent player already fully accounted for in current L5 — skip usage fallback too

    return {stat: delta * scale for stat, delta in raw_deltas.items()}


def _opponent_injury_delta(
    hist: pd.DataFrame,
    absent_name_lower: str,
    absent_team: str,
    active_name_lower: str,
    active_team: str,
    stat_cols: list[str],
) -> dict[str, float] | None:
    """Compute stat delta for an active player when an OPPONENT player is absent.

    Example: Siakam (IND) is OUT → Robinson (NYK) gets more boards vs. IND.

    Finds historical games where active_team played against absent_team,
    partitions them into games the absent player played vs. sat, and returns
    the mean stat difference for the active player.

    Returns None if the sample of absent games is too small.
    Returns {} if the absent player was already absent in the active player's
    recent matchups against this team (no adjustment needed).
    """
    # Active player's historical games vs absent player's team
    matchup_hist = hist[(hist["TEAM"] == active_team) & (hist["OPP"] == absent_team)]
    if matchup_hist.empty:
        return None

    all_matchup_games = set(matchup_hist["GAME_ID"].dropna())

    # Games where the absent player appeared for their team
    absent_in_opp = hist[
        (hist["TEAM"] == absent_team) &
        (hist["PLAYER_NAME"].str.lower() == absent_name_lower)
    ]
    games_with_absent = set(absent_in_opp["GAME_ID"].dropna())

    # Matchup games where the absent player sat
    games_absent_sat = all_matchup_games - games_with_absent

    if len(games_absent_sat) < _MIN_HISTORICAL_GAMES:
        return None

    active_matchup_rows = matchup_hist[
        matchup_hist["PLAYER_NAME"].str.lower() == active_name_lower
    ]
    if active_matchup_rows.empty:
        return None

    with_rows    = active_matchup_rows[active_matchup_rows["GAME_ID"].isin(games_with_absent)]
    without_rows = active_matchup_rows[active_matchup_rows["GAME_ID"].isin(games_absent_sat)]

    if len(without_rows) < _MIN_HISTORICAL_GAMES:
        return None

    # Scale by fraction of recent matchups where absent player was present
    # (same anti-double-counting logic as _historical_absent_delta)
    if "DATE" in active_matchup_rows.columns and "GAME_ID" in active_matchup_rows.columns:
        recent = (
            active_matchup_rows
            .dropna(subset=["DATE", "GAME_ID"])
            .sort_values("DATE", ascending=False)
            .head(5)["GAME_ID"]
            .tolist()
        )
        n_with = sum(1 for gid in recent if gid in games_with_absent)
        scale = n_with / max(len(recent), 1)
    else:
        scale = 1.0

    if scale <= 0:
        return {}   # already fully accounted for in recent matchup stats

    deltas: dict[str, float] = {}
    for stat in stat_cols:
        if stat not in hist.columns:
            continue
        mean_with    = with_rows[stat].mean()    if not with_rows.empty    else None
        mean_without = without_rows[stat].mean() if not without_rows.empty else None
        if mean_with is not None and mean_without is not None:
            deltas[stat] = (mean_without - mean_with) * scale

    return deltas if deltas else None


def _usage_redistribution_delta(
    absent_rows: pd.DataFrame,
    active_row: pd.Series,
    active_usage_total: float,
    stat_cols: list[str],
    df_cols: list[str],
) -> dict[str, float]:
    """Fallback: proportional usage redistribution when historical sample is too small."""
    player_usage = active_row.get("USAGE_PROXY_L5", 0) or 0
    if player_usage <= 0 or active_usage_total <= 0:
        return {}

    freed_usage = absent_rows["USAGE_PROXY_L5"].fillna(0).sum()
    share = player_usage / active_usage_total
    usage_boost = freed_usage * share

    deltas: dict[str, float] = {}
    for stat in stat_cols:
        col = f"{stat}_L5"
        if col not in df_cols:
            continue
        current_stat = active_row.get(col, 0) or 0
        if player_usage > 0 and current_stat > 0:
            stat_per_usage = current_stat / player_usage
            deltas[stat] = usage_boost * stat_per_usage
    return deltas


def apply_injury_adjustments(
    player_df: pd.DataFrame,
    injuries: pd.DataFrame,
    historical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adjust player stat projections based on today's OUT/DOUBTFUL teammates.

    Primary method — historical lookup (preferred):
      For each absent player X, find all past games where X's team played but X
      has no game-log row (he sat).  Compare each active teammate's actual stats
      in those games vs. games X played.  The observed mean difference is applied
      as the projection delta.

      This automatically handles the "already injured" case: if X has been out
      for the last 10 games, teammate Y's L5 stats already reflect X's absence,
      so the historical delta (which also averages X-out games) will be near-zero
      and no double-counting occurs.

    Fallback — usage redistribution (when sample < 5 absent games):
      Proportional usage reallocation translated to stat deltas via each player's
      recent stat-per-usage ratio.

    Args:
        player_df:     Most-recent player feature rows (one per player).
        injuries:      Today's injury report (PLAYER_NAME, TEAM, STATUS).
        historical_df: Full player_features.parquet (all seasons). If None,
                       falls back to usage redistribution for all players.

    Returns:
        player_df with {STAT}_L5 columns adjusted in-place for absent teammates.
        INJURY_ADJUSTED (bool) and INJURY_METHOD (str) columns mark which rows
        were modified and which method was used.
    """
    player_df = player_df.copy()
    player_df["INJURY_ADJUSTED"] = False
    player_df["INJURY_METHOD"] = ""

    if injuries.empty or player_df.empty:
        return player_df

    absent = injuries[injuries["STATUS"].isin(["OUT", "DOUBTFUL"])].copy()
    if absent.empty:
        return player_df

    # Raw stat columns we adjust (we read from actual game log columns, not _L5 derived ones,
    # for the historical delta; then apply the delta to the _L5 projection columns)
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN",
                 "PRA", "PR", "PA", "RA"]

    for team, absent_team in absent.groupby("TEAM"):
        team_mask = player_df["TEAM"] == team
        team_in_df = player_df[team_mask]
        if team_in_df.empty:
            continue

        absent_names_lower = set(absent_team["PLAYER_NAME"].str.lower())
        active_mask  = ~team_in_df["PLAYER_NAME"].str.lower().isin(absent_names_lower)
        absent_rows  = team_in_df[~active_mask]
        active_rows  = team_in_df[active_mask]

        if active_rows.empty:
            continue

        active_usage_total = active_rows["USAGE_PROXY_L5"].fillna(0).sum()

        # Log who is out
        for _, absent_row in absent_team.iterrows():
            match = absent_rows[absent_rows["PLAYER_NAME"].str.lower() == absent_row["PLAYER_NAME"].lower()]
            usage_str = f"usage={match['USAGE_PROXY_L5'].values[0]:.1f}" if not match.empty else "usage=unknown"
            log.info("  OUT/DOUBTFUL: %s (%s, %s) — %s",
                     absent_row["PLAYER_NAME"], team, absent_row["STATUS"], usage_str)

        for idx, active_row in active_rows.iterrows():
            teammate_name_lower = active_row["PLAYER_NAME"].lower()
            cumulative_deltas: dict[str, float] = {}

            for _, absent_row in absent_team.iterrows():
                absent_name_lower = absent_row["PLAYER_NAME"].lower()

                # Try historical lookup first.
                # Returns: dict with deltas → use them
                #          {}   (empty dict) → absent player already in L5, skip usage fallback
                #          None → insufficient sample, fall back to usage redistribution
                method = "usage"
                deltas: dict[str, float] | None = None

                if historical_df is not None:
                    hist_result = _historical_absent_delta(
                        historical_df, absent_name_lower, teammate_name_lower, team, stat_cols
                    )
                    if hist_result is not None:
                        deltas = hist_result
                        method = "historical" if hist_result else "already-adjusted"

                if deltas is None:
                    # Historical had no sample — fall back to usage redistribution
                    deltas = _usage_redistribution_delta(
                        absent_rows[absent_rows["PLAYER_NAME"].str.lower() == absent_name_lower],
                        active_row,
                        active_usage_total,
                        stat_cols,
                        list(player_df.columns),
                    )

                for stat, delta in deltas.items():
                    cumulative_deltas[stat] = cumulative_deltas.get(stat, 0) + delta

                if deltas:
                    log.debug(
                        "  %s without %s (%s): %s",
                        active_row["PLAYER_NAME"], absent_row["PLAYER_NAME"], method,
                        ", ".join(f"{s}={v:+.2f}" for s, v in deltas.items() if abs(v) >= 0.1)
                    )

            # Store injury deltas in separate columns; do NOT mutate _L5 so the
            # model always receives the features it was trained on.
            if cumulative_deltas:
                for stat, delta in cumulative_deltas.items():
                    delta_col = f"{stat}_INJ_DELTA"
                    if delta_col not in player_df.columns:
                        player_df[delta_col] = 0.0
                    player_df.at[idx, delta_col] = player_df.at[idx, delta_col] + delta

                methods_used = set()
                for _, ar in absent_team.iterrows():
                    if historical_df is not None:
                        hd = _historical_absent_delta(
                            historical_df, ar["PLAYER_NAME"].lower(),
                            teammate_name_lower, team, stat_cols,
                        )
                        if hd is None:
                            methods_used.add("usage")
                        elif hd:
                            methods_used.add("historical")
                        else:
                            methods_used.add("already-adjusted")
                    else:
                        methods_used.add("usage")

                player_df.at[idx, "INJURY_ADJUSTED"] = True
                player_df.at[idx, "INJURY_METHOD"] = "+".join(sorted(methods_used))

    # -----------------------------------------------------------------------
    # Part 2: OPPONENT injuries → cross-team effects
    # E.g., if Siakam (IND) is out, NYK's Robinson gets more boards vs IND.
    # Requires that player_df["OPP"] is set to TODAY's opponent (done in main()).
    # -----------------------------------------------------------------------
    if historical_df is not None and "OPP" in player_df.columns:
        for idx, active_row in player_df.iterrows():
            active_team = active_row.get("TEAM", "")
            opp_team    = active_row.get("OPP", "")
            if not active_team or not opp_team:
                continue

            # Absent players from the opposing team
            absent_opp = absent[absent["TEAM"] == opp_team]
            if absent_opp.empty:
                continue

            active_name_lower = active_row["PLAYER_NAME"].lower()
            opp_cumulative: dict[str, float] = {}
            opp_methods: set[str] = set()

            for _, absent_row in absent_opp.iterrows():
                absent_name_lower = absent_row["PLAYER_NAME"].lower()
                result = _opponent_injury_delta(
                    historical_df,
                    absent_name_lower,
                    opp_team,
                    active_name_lower,
                    active_team,
                    stat_cols,
                )
                if result is None:
                    opp_methods.add("opp-no-sample")
                    continue
                if not result:
                    opp_methods.add("opp-already-adjusted")
                    continue

                for stat, delta in result.items():
                    opp_cumulative[stat] = opp_cumulative.get(stat, 0) + delta
                opp_methods.add("opp-historical")

                if result:
                    log.info(
                        "  %s vs %s (without %s): %s",
                        active_row["PLAYER_NAME"], opp_team, absent_row["PLAYER_NAME"],
                        ", ".join(f"{s}={v:+.2f}" for s, v in result.items() if abs(v) >= 0.1),
                    )

            if opp_cumulative:
                for stat, delta in opp_cumulative.items():
                    delta_col = f"{stat}_OPP_INJ_DELTA"
                    if delta_col not in player_df.columns:
                        player_df[delta_col] = 0.0
                    player_df.at[idx, delta_col] = player_df.at[idx, delta_col] + delta

                player_df.at[idx, "INJURY_ADJUSTED"] = True
                existing_method = player_df.at[idx, "INJURY_METHOD"]
                all_methods = set(existing_method.split("+")) | opp_methods if existing_method else opp_methods
                player_df.at[idx, "INJURY_METHOD"] = "+".join(sorted(m for m in all_methods if m))

    n_adjusted = int(player_df["INJURY_ADJUSTED"].sum())
    if n_adjusted > 0:
        log.info("Injury adjustment applied to %d player(s).", n_adjusted)
        # Log summary of changes for players who have picks
        for _, row in player_df[player_df["INJURY_ADJUSTED"]].iterrows():
            log.info(
                "  %-25s (%s, %s): REB_L5=%s  AST_L5=%s  PTS_L5=%s  PRA_L5=%s",
                row["PLAYER_NAME"], row["TEAM"], row["INJURY_METHOD"],
                f"{row['REB_L5']:.1f}" if "REB_L5" in player_df.columns else "n/a",
                f"{row['AST_L5']:.1f}" if "AST_L5" in player_df.columns else "n/a",
                f"{row['PTS_L5']:.1f}" if "PTS_L5" in player_df.columns else "n/a",
                f"{row['PRA_L5']:.1f}" if "PRA_L5" in player_df.columns else "n/a",
            )
    else:
        log.info("Injury adjustments: no changes applied (absent players not found in feature data or sample too small).")

    return player_df


_MATCHUP_FACTOR_CAP   = 1.20   # max multiplier in either direction
_MATCHUP_FACTOR_FLOOR = 1 / _MATCHUP_FACTOR_CAP
_MATCHUP_MIN_DELTA    = 0.03   # only apply if factor deviates > 3% from 1.0
_MATCHUP_RECENT_SEASONS = 2    # how many most-recent seasons to use


def apply_matchup_adjustments(
    player_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Scale player L5 stat projections by opponent's defensive quality factor.

    For each player P facing today's opponent T, for each stat S:
      factor = mean(S allowed per player-game vs T, recent seasons)
               / mean(S per player-game, league-wide, recent seasons)

    A factor > 1 means T gives up more of stat S than average — players facing
    T should see a boost.  A factor < 1 means T is stingy with that stat.

    The factor naturally amplifies for high-volume players: a center with
    REB_L5 = 10 gets a bigger absolute bump than a guard with REB_L5 = 2,
    even though the same multiplier is applied.

    Uses the most recent `_MATCHUP_RECENT_SEASONS` seasons to avoid stale team
    personnel/scheme effects.  Factor is capped at [0.83, 1.20] to prevent
    extreme adjustments from small-sample outlier teams.

    Requires player_df["OPP"] to reflect TODAY's opponent (set in main() from
    game_df before calling this function).

    Args:
        player_df:     Most-recent player feature rows (one per player, with OPP).
        historical_df: Full player-game history for factor computation.

    Returns:
        player_df with L5 stat columns scaled.
        MATCHUP_ADJUSTED (bool), MATCHUP_FACTORS (str summary) columns added.
    """
    player_df = player_df.copy()
    player_df["MATCHUP_ADJUSTED"] = False
    player_df["MATCHUP_FACTORS"] = ""

    if historical_df.empty or "OPP" not in player_df.columns:
        return player_df

    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "PRA", "PR", "PA", "RA"]

    # Use only the most recent N seasons for relevance
    if "SEASON" in historical_df.columns:
        recent_seasons = sorted(historical_df["SEASON"].unique())[-_MATCHUP_RECENT_SEASONS:]
        recent_hist = historical_df[historical_df["SEASON"].isin(recent_seasons)]
    else:
        recent_hist = historical_df

    if recent_hist.empty:
        return player_df

    # League-wide mean per stat (denominator)
    league_means: dict[str, float] = {}
    for stat in stat_cols:
        if stat in recent_hist.columns:
            m = recent_hist[stat].mean()
            if m and m > 0:
                league_means[stat] = m

    if not league_means:
        return player_df

    # Pre-compute per-opponent defensive factors (once for all 30 teams)
    opp_factors: dict[str, dict[str, float]] = {}   # opp_team → {stat → factor}
    if "OPP" in recent_hist.columns:
        for opp_team, opp_group in recent_hist.groupby("OPP"):
            factors: dict[str, float] = {}
            for stat, league_mean in league_means.items():
                if stat not in opp_group.columns:
                    continue
                opp_mean = opp_group[stat].mean()
                if opp_mean > 0 and league_mean > 0:
                    raw_factor = opp_mean / league_mean
                    factors[stat] = max(_MATCHUP_FACTOR_FLOOR, min(_MATCHUP_FACTOR_CAP, raw_factor))
            opp_factors[str(opp_team)] = factors

    if not opp_factors:
        return player_df

    for idx, row in player_df.iterrows():
        opp_team = str(row.get("OPP", "") or "")
        if not opp_team or opp_team not in opp_factors:
            continue

        factors = opp_factors[opp_team]
        applied: list[str] = []

        for stat, factor in factors.items():
            if abs(factor - 1.0) < _MATCHUP_MIN_DELTA:
                continue   # within noise band — skip

            col = f"{stat}_L5"
            if col not in player_df.columns:
                continue

            current = player_df.at[idx, col]
            if not current or current <= 0:
                continue

            factor_col = f"{stat}_MATCHUP_FACTOR"
            if factor_col not in player_df.columns:
                player_df[factor_col] = 1.0
            player_df.at[idx, factor_col] = factor
            applied.append(f"{stat}x{factor:.2f}")

        if applied:
            player_df.at[idx, "MATCHUP_ADJUSTED"] = True
            player_df.at[idx, "MATCHUP_FACTORS"] = " ".join(applied)

    n_adj = int(player_df["MATCHUP_ADJUSTED"].sum())
    if n_adj > 0:
        log.info("Matchup adjustments applied to %d player(s) (opp def factors).", n_adj)

    return player_df


def _warn_injured_picks(picks, injuries: pd.DataFrame) -> None:
    """Log a warning if any pick involves a teammate of an OUT/DOUBTFUL player.

    This is an informational alert — the model may not have accounted for
    injuries that occurred after the player features were last built.
    """
    if injuries.empty:
        return

    from src.betting.edge_calculator import PropPick
    critical = injuries[injuries["STATUS"].isin(["OUT", "DOUBTFUL"])]
    if critical.empty:
        return

    # Build: team → set of out players
    out_by_team: dict[str, list[str]] = {}
    for _, row in critical.iterrows():
        out_by_team.setdefault(row["TEAM"], []).append(f"{row['PLAYER_NAME']} ({row['STATUS']})")

    warned: set[str] = set()
    for pick in picks:
        if not isinstance(pick, PropPick):
            continue
        team_out = out_by_team.get(pick.team, [])
        if team_out:
            key = f"{pick.player_name}|{pick.team}"
            if key not in warned:
                warned.add(key)
                log.warning(
                    "INJURY ALERT: %s (%s) — teammate(s) OUT/DOUBTFUL: %s",
                    pick.player_name, pick.team, ", ".join(team_out),
                )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_pick(
    pick,
    injury_adjusted_players: set[str] | None = None,
    injury_method_map: dict[str, str] | None = None,
) -> str:
    from src.betting.edge_calculator import GamePick, PropPick
    if isinstance(pick, PropPick):
        adjusted = (injury_adjusted_players or set())
        if pick.player_name in adjusted:
            method = (injury_method_map or {}).get(pick.player_name, "")
            tag = f" [INJ-ADJ:{method}]" if method else " [INJ-ADJ]"
        else:
            tag = ""
        return (
            f"  {pick.player_name} ({pick.team} vs {pick.opp}) "
            f"{pick.stat} {pick.direction.upper()} {pick.line:g}  "
            f"[model={pick.model_median:.1f}, edge={pick.edge:+.1%}]{tag}"
        )
    if isinstance(pick, GamePick):
        team = pick.home_team if pick.direction == "home" else pick.away_team
        return (
            f"  {team} to WIN ({pick.home_team} vs {pick.away_team})  "
            f"[prob={pick.model_prob_home:.1%}, edge={pick.edge:+.1%}]"
        )
    return str(pick)


def print_pick_sheet(
    entries: list,
    sizing_df: pd.DataFrame,
    bankroll: float,
    today: date,
    injuries: pd.DataFrame | None = None,
    injury_adjusted_players: set[str] | None = None,
    injury_method_map: dict[str, str] | None = None,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  UNDERDOG FANTASY PICKS — {today}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(sep)

    # Print injury summary block
    if injuries is not None and not injuries.empty:
        critical = injuries[injuries["STATUS"].isin(["OUT", "DOUBTFUL"])]
        if not critical.empty:
            print(f"\n{'-' * 60}")
            print("  INJURY ALERT -- OUT / DOUBTFUL PLAYERS TODAY")
            print(f"{'-' * 60}")
            for _, row in critical.iterrows():
                print(f"  {row['TEAM']:<4}  {row['PLAYER_NAME']:<25}  {row['STATUS']:<12}  {row['REASON']}")
            print()

    ranked = rank_entries(entries)
    for i, (picks, score) in enumerate(ranked, 1):
        n         = len(picks)
        payout    = UNDERDOG_PAYOUTS.get(n, 0)
        row       = sizing_df.iloc[i - 1] if i <= len(sizing_df) else None
        amount    = row["bet_amount"] if row is not None else 0.0
        win_prob  = row["win_prob"]   if row is not None else 0.0
        ev        = row["ev"]         if row is not None else 0.0

        print(f"\nEntry {i}: {n}-pick (payout {payout:.0f}×)  "
              f"win_prob={win_prob:.1%}  EV={ev:+.2f}  "
              f"Bet: ${amount:.2f}")
        for pick in picks:
            print(_format_pick(pick, injury_adjusted_players, injury_method_map))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Post-hoc prediction adjustment
# ---------------------------------------------------------------------------

def _filter_priority_picks(picks: list, player_df: pd.DataFrame) -> list:
    """Keep only priority stats; apply role filters for AST and REB.

    Priority stats: PTS, PRA, AST (primary ball-handlers), FG3M, REB (centers).
    Everything else (STL, BLK, TOV, PR, PA, RA, etc.) is silently dropped
    until explicitly re-enabled via _PRIORITY_STATS.
    """
    # Build player → L10 lookup from adjusted player_df (post injury/matchup)
    l10_lookup: dict[tuple[str, str], float] = {}
    for _, row in player_df.iterrows():
        name = str(row.get("PLAYER_NAME", ""))
        for stat in ("AST", "REB"):
            col = f"{stat}_L10"
            if col in row.index and pd.notna(row[col]):
                l10_lookup[(name, stat)] = float(row[col])

    kept = []
    for pick in picks:
        stat = getattr(pick, "stat", None)
        if stat not in _PRIORITY_STATS:
            continue

        if stat == "AST":
            l10 = l10_lookup.get((pick.player_name, "AST"), 0.0)
            if l10 < _AST_PRIMARY_HANDLER_L10:
                continue  # not a primary ball-handler

        if stat == "REB":
            l10 = l10_lookup.get((pick.player_name, "REB"), 0.0)
            if l10 < _REB_CENTER_L10:
                continue  # not a center/big

        kept.append(pick)

    return kept


def _post_hoc_adjust_picks(
    picks: list,
    player_df: pd.DataFrame,
    min_edge: float,
) -> list:
    """Adjust model median/interval using stored INJ_DELTA and MATCHUP_FACTOR columns.

    The model always runs on raw features (trained distribution).  Injury and
    matchup adjustments are stored as separate delta/factor columns and applied
    here to the model's OUTPUT, keeping input features intact.
    """
    from src.betting.edge_calculator import calculate_prop_edge, prob_over_from_quantiles
    from dataclasses import replace as dc_replace

    if player_df.empty:
        return picks

    name_idx = player_df.set_index("PLAYER_NAME") if "PLAYER_NAME" in player_df.columns else None

    adjusted = []
    for pick in picks:
        try:
            row = name_idx.loc[pick.player_name] if name_idx is not None else None
            if row is None or (isinstance(row, pd.DataFrame) and row.empty):
                adjusted.append(pick)
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            stat = pick.stat
            # Teammate injury delta: capped at ±25% of raw prediction
            # (strong signal when star is out, but still bounded)
            inj_delta      = float(row.get(f"{stat}_INJ_DELTA",     0.0) or 0.0)
            # Opponent injury delta: capped at ±10% — small-sample noise is high
            opp_inj_delta  = float(row.get(f"{stat}_OPP_INJ_DELTA", 0.0) or 0.0)
            matchup_factor = float(row.get(f"{stat}_MATCHUP_FACTOR", 1.0) or 1.0)

            if inj_delta == 0.0 and opp_inj_delta == 0.0 and matchup_factor == 1.0:
                adjusted.append(pick)
                continue

            raw_median = pick.model_median
            if raw_median > 0:
                inj_delta     = max(-raw_median * 0.25, min(raw_median * 0.25, inj_delta))
                opp_inj_delta = max(-raw_median * 0.10, min(raw_median * 0.10, opp_inj_delta))

            total_delta = inj_delta + opp_inj_delta
            new_median = max(0.0, (raw_median + total_delta) * matchup_factor)
            # Shift interval by same absolute delta as median was shifted
            shift = new_median - raw_median
            new_low  = max(0.0, pick.model_low  + shift)
            new_high = max(0.0, pick.model_high + shift)

            new_prob_over = prob_over_from_quantiles(new_median, new_low, new_high, pick.line)
            new_edge      = new_prob_over - pick.underdog_prob_over

            pick.model_median    = round(new_median, 2)
            pick.model_low       = round(new_low, 2)
            pick.model_high      = round(new_high, 2)
            pick.model_prob_over = round(new_prob_over, 4)
            pick.edge            = round(new_edge, 4)

            if abs(new_edge) >= min_edge:
                adjusted.append(pick)
        except Exception:
            adjusted.append(pick)

    return adjusted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Underdog pick pipeline.")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--bankroll", type=float, default=_DEFAULT_BANKROLL,
                        help=f"Current bankroll in dollars (default: {_DEFAULT_BANKROLL})")
    parser.add_argument("--min-edge", type=float, default=_DEFAULT_MIN_EDGE,
                        help=f"Minimum edge to consider (default: {_DEFAULT_MIN_EDGE})")
    parser.add_argument("--max-entries", type=int, default=_DEFAULT_MAX_ENTRIES,
                        help=f"Maximum entries to output (default: {_DEFAULT_MAX_ENTRIES})")
    parser.add_argument("--kelly", type=float, default=_DEFAULT_KELLY,
                        help=f"Kelly fraction (default: {_DEFAULT_KELLY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip DB writes and Underdog API calls")
    args = parser.parse_args()

    today: date = (
        date.fromisoformat(args.date) if args.date
        else date.today()
    )

    log.info("=" * 60)
    log.info("Daily pipeline — %s (dry_run=%s)", today, args.dry_run)
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Fetch Underdog lines (always — read-only call)
    #    --dry-run only skips DB writes and entry logging, not the fetch.
    # -----------------------------------------------------------------------
    prop_lines = []
    game_lines = []

    try:
        prop_lines = fetch_prop_lines(str(today))
        game_lines = fetch_game_lines(str(today))
        if not args.dry_run:
            save_lines_to_db(prop_lines, game_lines)
        else:
            log.info("--dry-run: fetched %d prop lines (DB write skipped).", len(prop_lines))
    except UnderdogAuthError as exc:
        log.error("Underdog auth failed: %s", exc)
        log.error("The public endpoint returned 401 — endpoint may have changed.")
        sys.exit(1)
    except Exception as exc:
        log.warning("Could not fetch Underdog lines: %s — continuing with empty pool.", exc)

    # Detect and log combo line divergences (PRA / RA vs component sums).
    # These are logged every run for historical tracking; actual exploitation
    # of the edge is validated by analyze_line_divergence.py after settlement.
    if prop_lines:
        try:
            divergences = detect_divergences(prop_lines)
            if divergences:
                log.info(
                    "Combo line divergences (>= %.1f pts): %d found.",
                    1.5, len(divergences),
                )
                for d in divergences:
                    direction = "OVER" if d.divergence > 0 else "UNDER"
                    comp_str  = "+".join(f"{k}={v}" for k, v in d.component_lines.items())
                    log.info(
                        "  %s %s %s %.1f  (sum=%s -> %s %.1f, div=%+.1f, edge~%+.1fpp)",
                        d.player_name, d.combo_stat, direction, d.combo_line,
                        comp_str, direction, d.sum_individual, d.divergence, d.edge_pp,
                    )
                if not args.dry_run:
                    log_divergences_to_db(divergences)
        except Exception as exc:
            log.warning("Divergence detection failed: %s", exc)

    # -----------------------------------------------------------------------
    # 2. Load features and models
    # -----------------------------------------------------------------------
    game_df   = load_today_game_features(today)
    player_df = load_today_player_features(today)
    injuries  = load_injury_report()

    # Load full history for injury delta lookups and matchup factors
    historical_df: pd.DataFrame | None = None
    if _PLAYER_FEATURES_PATH.exists():
        try:
            historical_df = pd.read_parquet(_PLAYER_FEATURES_PATH)
            log.info("Loaded %d historical player-game rows for injury/matchup analysis.", len(historical_df))
        except Exception as exc:
            log.warning("Could not load full player history for injury/matchup analysis: %s", exc)

    # Update OPP column to reflect today's actual opponent (not last game's opponent)
    if not game_df.empty and "HOME" in game_df.columns and "AWAY" in game_df.columns:
        today_team_to_opp: dict[str, str] = {}
        for _, game_row in game_df.iterrows():
            today_team_to_opp[str(game_row["HOME"])] = str(game_row["AWAY"])
            today_team_to_opp[str(game_row["AWAY"])] = str(game_row["HOME"])
        if today_team_to_opp and "TEAM" in player_df.columns:
            player_df["OPP"] = player_df["TEAM"].map(today_team_to_opp).fillna(
                player_df["OPP"] if "OPP" in player_df.columns else ""
            )
            log.info("Updated OPP to today's matchups for %d players.", player_df["OPP"].notna().sum())

    # Save raw features BEFORE adjustments — model must receive features from
    # its training distribution, not injury/matchup-modified values.
    player_df_raw = player_df.copy()

    player_df = apply_injury_adjustments(player_df, injuries, historical_df=historical_df)
    if historical_df is not None:
        player_df = apply_matchup_adjustments(player_df, historical_df)
    game_model = load_game_model()
    prop_models = load_prop_models()

    # -----------------------------------------------------------------------
    # 3. Screen picks
    # -----------------------------------------------------------------------
    all_picks = []

    if game_model and not game_df.empty and game_lines:
        try:
            game_df_prep = prepare_features(game_df.copy())
            game_picks   = screen_game_picks(game_df_prep, game_model, game_lines,
                                             min_edge=args.min_edge)
            all_picks.extend(game_picks)
            log.info("Game picks: %d with edge >= %.2f.", len(game_picks), args.min_edge)
        except Exception as exc:
            log.warning("Game pick screening failed: %s", exc)

    if prop_models and not player_df_raw.empty and prop_lines:
        try:
            # Run model on RAW features (unmodified by injury/matchup adjustments)
            prop_picks = screen_prop_picks(player_df_raw, prop_models, prop_lines,
                                           min_edge=args.min_edge)
            # Post-hoc: shift predictions by injury deltas and matchup factors
            prop_picks = _post_hoc_adjust_picks(prop_picks, player_df, args.min_edge)
            # Restrict to priority stats (PTS, PRA, AST for PGs, FG3M, REB for bigs)
            prop_picks = _filter_priority_picks(prop_picks, player_df)
            all_picks.extend(prop_picks)
            log.info("Prop picks: %d with edge >= %.2f (priority stats only).",
                     len(prop_picks), args.min_edge)
        except Exception as exc:
            log.warning("Prop pick screening failed: %s", exc)

    if not all_picks:
        log.info("No positive-edge picks found for %s — no entries built.", today)
        print(f"\nNo picks with edge >= {args.min_edge:.0%} found for {today}.")
        return

    # Warn about injured teammates before building entries
    _warn_injured_picks(all_picks, injuries)

    # -----------------------------------------------------------------------
    # 4. Build entries
    # -----------------------------------------------------------------------
    entries = build_entries(
        all_picks,
        min_picks=2,
        max_picks=5,
        max_entries=args.max_entries,
    )

    if not entries:
        log.info("No entries built (need at least 2 picks).")
        return

    # -----------------------------------------------------------------------
    # 5. Size entries
    # -----------------------------------------------------------------------
    sizing_df = summarise_sizing(entries, bankroll=args.bankroll, kelly_fraction=args.kelly)

    # -----------------------------------------------------------------------
    # 6. Print pick sheet
    # -----------------------------------------------------------------------
    # Build map: player_name → adjustment method for pick sheet annotation
    injury_adjusted_players: set[str] = set()
    injury_method_map: dict[str, str] = {}
    if "INJURY_ADJUSTED" in player_df.columns:
        adj_rows = player_df[player_df["INJURY_ADJUSTED"]]
        injury_adjusted_players = set(adj_rows["PLAYER_NAME"].tolist())
        if "INJURY_METHOD" in player_df.columns:
            for _, row in adj_rows.iterrows():
                injury_method_map[row["PLAYER_NAME"]] = row.get("INJURY_METHOD", "")

    print_pick_sheet(
        entries, sizing_df,
        bankroll=args.bankroll,
        today=today,
        injuries=injuries,
        injury_adjusted_players=injury_adjusted_players,
        injury_method_map=injury_method_map,
    )

    # -----------------------------------------------------------------------
    # 7. Log to DB (unless dry run)
    # -----------------------------------------------------------------------
    if not args.dry_run:
        for i, (picks, _score) in enumerate(rank_entries(entries)):
            row    = sizing_df.iloc[i] if i < len(sizing_df) else None
            amount = float(row["bet_amount"]) if row is not None else 0.0
            if amount <= 0:
                continue
            ref = log_entry(picks, bet_amount=amount, game_date=today)
            log.info("Logged entry %s ($%.2f).", ref[:8], amount)

    log.info("Done.")


if __name__ == "__main__":
    main()

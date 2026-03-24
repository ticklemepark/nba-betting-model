"""Rolling player stat averages for NBA player prop modeling.

For each player-game in `player_games`, computes rolling averages of any
requested stat (PTS, REB, AST, STL, BLK, TOV, FG3M, MIN, etc.) over
the last L5 / L10 / L20 games BEFORE the target game date.

Design mirrors the team rolling modules: per-player chronological sort,
rolling mean through each game, then merge_asof to attach pre-game stats.

Usage:
    from src.data.nba_api_client import fetch_player_game_logs
    from src.features.player.rolling_stats import compute_rolling_player_stats

    logs = fetch_player_game_logs(2024)
    # player_games: PLAYER_ID, TEAM, OPP, DATE, IS_HOME per upcoming prop
    player_games = compute_rolling_player_stats(
        logs, player_games,
        stats=["PTS", "REB", "AST", "FG3M", "MIN"],
        windows=[5, 10, 20],
    )
    # Adds: PTS_L5, PTS_L10, PTS_L20, REB_L5 ... MIN_L20
    # Also adds: PTS_SEASON, REB_SEASON, ... (full season average)
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_STATS: list[str] = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
DEFAULT_WINDOWS: list[int] = [5, 10, 20]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_player_stats(
    player_game_logs: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str] = DEFAULT_STATS,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach rolling stat columns to a player_games DataFrame.

    Args:
        player_game_logs: Output of fetch_player_game_logs(). Must contain
            PLAYER_ID, DATE, SEASON, and all columns listed in `stats`.
        player_games: One row per player per upcoming game. Must contain
            PLAYER_ID, DATE.
        stats: Stat columns to compute rolling averages for.
        windows: Rolling window sizes. Defaults to [5, 10, 20].

    Returns:
        Copy of player_games with added columns per stat per window:
            {STAT}_L{W}     - rolling mean of last W games
            {STAT}_SEASON   - season-to-date mean (all games before game date)

        NaN where a player has no prior games.
    """
    _validate_logs(player_game_logs, stats)
    _validate_player_games(player_games)

    rolling = _compute_rolling(player_game_logs, stats, windows)
    return _attach_to_player_games(rolling, player_games, stats, windows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_rolling(
    logs: pd.DataFrame,
    stats: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Build per-player rolling stat table.

    For each player, computes rolling means (through each game, inclusive)
    so that merge_asof with direction='backward' gives pre-game stats.
    Also computes an expanding season mean, reset at each new season.
    """
    logs = logs.copy()
    logs["DATE"] = pd.to_datetime(logs["DATE"])

    parts = []
    for player_id, grp in logs.groupby("PLAYER_ID"):
        grp = grp.sort_values("DATE").copy()
        for w in windows:
            for stat in stats:
                if stat in grp.columns:
                    grp[f"{stat}_L{w}"] = (
                        grp[stat].rolling(window=w, min_periods=1).mean()
                    )
        # Season-to-date mean: expanding within each season
        for season_val, season_grp in grp.groupby("SEASON"):
            idx = season_grp.index
            for stat in stats:
                if stat in grp.columns:
                    grp.loc[idx, f"{stat}_SEASON"] = (
                        season_grp[stat].expanding(min_periods=1).mean().values
                    )
        parts.append(grp)

    if not parts:
        return logs

    result = pd.concat(parts).reset_index(drop=True)
    return result


def _attach_to_player_games(
    rolling: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Join rolling player stats to player_games via merge_asof on (PLAYER_ID, DATE)."""
    player_games = player_games.copy()
    player_games["DATE"] = pd.to_datetime(player_games["DATE"])
    player_games = player_games.sort_values("DATE").reset_index(drop=True)

    feature_cols = (
        [f"{s}_L{w}" for s in stats for w in windows if f"{s}_L{w}" in rolling.columns]
        + [f"{s}_SEASON" for s in stats if f"{s}_SEASON" in rolling.columns]
    )

    rolling_slim = rolling[["PLAYER_ID", "DATE"] + feature_cols].sort_values("DATE")

    player_games["PLAYER_ID"] = player_games["PLAYER_ID"].astype(object)
    rolling_slim["PLAYER_ID"] = rolling_slim["PLAYER_ID"].astype(object)
    result = pd.merge_asof(
        player_games,
        rolling_slim,
        on="DATE",
        by="PLAYER_ID",
        direction="backward",
    )
    return result


def _validate_logs(logs: pd.DataFrame, stats: list[str]) -> None:
    required = {"PLAYER_ID", "DATE", "SEASON"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"player_game_logs missing columns: {sorted(missing)}")
    missing_stats = [s for s in stats if s not in logs.columns]
    if missing_stats:
        log.warning("Stats not found in player_game_logs: %s", missing_stats)


def _validate_player_games(player_games: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")

"""Home/away stat split features for NBA players.

Some players perform significantly differently at home vs. away — particularly
for 3-point shooting and points.  These splits are computed season-to-date
(not rolling window) to maintain sufficient sample sizes.

Usage:
    from src.features.player.home_away import compute_home_away_splits

    player_games = compute_home_away_splits(
        player_log, player_games,
        stats=["PTS", "REB", "AST", "FG3M"],
    )
    # Adds: PTS_HOME_AVG, PTS_AWAY_AVG, PTS_HOME_AWAY_DIFF
    #       REB_HOME_AVG, ...
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_STATS: list[str] = ["PTS", "REB", "AST", "FG3M", "MIN"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_home_away_splits(
    player_game_logs: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str] = DEFAULT_STATS,
) -> pd.DataFrame:
    """Attach season-to-date home/away split columns to player_games.

    For each player-game in player_games, looks up the player's home and away
    averages using only games played BEFORE the target date (no leakage).

    Args:
        player_game_logs: Output of fetch_player_game_logs(). Must contain
            PLAYER_ID, DATE, SEASON, IS_HOME, and all stats.
        player_games: One row per player per game. Must contain
            PLAYER_ID, DATE, SEASON.
        stats: Stats to compute splits for.

    Returns:
        Copy of player_games with added columns per stat:
            {STAT}_HOME_AVG       - season-to-date avg in home games
            {STAT}_AWAY_AVG       - season-to-date avg in away games
            {STAT}_HOME_AWAY_DIFF - HOME_AVG minus AWAY_AVG (positive = better at home)

        NaN when a player has no home or away games yet this season.
    """
    _validate_logs(player_game_logs, stats)
    _validate_player_games(player_games)

    home_rolling, away_rolling = _compute_split_rolling(player_game_logs, stats)
    return _attach_to_player_games(home_rolling, away_rolling, player_games, stats)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_split_rolling(
    logs: pd.DataFrame,
    stats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute season-to-date expanding means split by IS_HOME, per player."""
    logs = logs.copy()
    logs["DATE"] = pd.to_datetime(logs["DATE"])

    home_parts, away_parts = [], []

    for _, grp in logs.groupby("PLAYER_ID"):
        grp = grp.sort_values("DATE").copy()

        home_grp = grp[grp["IS_HOME"] == True].copy()
        away_grp = grp[grp["IS_HOME"] == False].copy()

        for season_val, sg in home_grp.groupby("SEASON"):
            idx = sg.index
            for stat in stats:
                if stat in sg.columns:
                    home_grp.loc[idx, f"{stat}_HOME_AVG"] = (
                        sg[stat].expanding(min_periods=1).mean().values
                    )

        for season_val, sg in away_grp.groupby("SEASON"):
            idx = sg.index
            for stat in stats:
                if stat in sg.columns:
                    away_grp.loc[idx, f"{stat}_AWAY_AVG"] = (
                        sg[stat].expanding(min_periods=1).mean().values
                    )

        home_parts.append(home_grp)
        away_parts.append(away_grp)

    home_rolling = pd.concat(home_parts).reset_index(drop=True) if home_parts else logs.iloc[:0]
    away_rolling = pd.concat(away_parts).reset_index(drop=True) if away_parts else logs.iloc[:0]
    return home_rolling, away_rolling


def _attach_to_player_games(
    home_rolling: pd.DataFrame,
    away_rolling: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str],
) -> pd.DataFrame:
    player_games = player_games.copy()
    player_games["DATE"] = pd.to_datetime(player_games["DATE"])
    player_games = player_games.sort_values("DATE").reset_index(drop=True)

    home_cols = [f"{s}_HOME_AVG" for s in stats if f"{s}_HOME_AVG" in home_rolling.columns]
    away_cols = [f"{s}_AWAY_AVG" for s in stats if f"{s}_AWAY_AVG" in away_rolling.columns]

    if home_cols and not home_rolling.empty:
        home_slim = home_rolling[["PLAYER_ID", "DATE"] + home_cols].sort_values("DATE")
        player_games = pd.merge_asof(
            player_games, home_slim, on="DATE", by="PLAYER_ID", direction="backward"
        )

    if away_cols and not away_rolling.empty:
        away_slim = away_rolling[["PLAYER_ID", "DATE"] + away_cols].sort_values("DATE")
        player_games = pd.merge_asof(
            player_games, away_slim, on="DATE", by="PLAYER_ID", direction="backward"
        )

    # Add diff columns where both splits exist
    for stat in stats:
        h = f"{stat}_HOME_AVG"
        a = f"{stat}_AWAY_AVG"
        if h in player_games.columns and a in player_games.columns:
            player_games[f"{stat}_HOME_AWAY_DIFF"] = player_games[h] - player_games[a]

    return player_games


def _validate_logs(logs: pd.DataFrame, stats: list[str]) -> None:
    required = {"PLAYER_ID", "DATE", "SEASON", "IS_HOME"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"player_game_logs missing columns: {sorted(missing)}")


def _validate_player_games(player_games: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")

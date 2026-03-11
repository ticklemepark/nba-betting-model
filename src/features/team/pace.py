"""Rolling pace features for NBA teams.

Pace measures how many possessions a team plays per 48 minutes.  High-pace
teams generate more possessions and therefore more raw counting stats for
all players — a critical context feature for player prop modelling.

Pace formula (per game):
    possessions = FGA - OREB + TOV + 0.44 * FTA
    pace        = possessions / game_minutes * 48

For regulation games game_minutes = 48; OT adds 5 min per period.
The MIN column from nba_api is the game duration in minutes.

Projected game pace = (home_pace_L{W} + away_pace_L{W}) / 2
This predicts the overall tempo of the upcoming game.

Usage:
    from src.data.nba_api_client import fetch_team_game_logs
    from src.features.team.pace import compute_rolling_pace

    logs  = fetch_team_game_logs(2024)
    games = compute_rolling_pace(logs, games)
    # Adds: HOME_PACE_L5/L10/L20, AWAY_PACE_L5/L10/L20,
    #       PROJ_PACE_L5/L10/L20
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_WINDOWS: list[int] = [5, 10, 20]
_REGULATION_MINUTES = 48.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_pace(
    team_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach rolling pace columns to a games DataFrame.

    Args:
        team_game_logs: Output of fetch_team_game_logs(). Must contain
            TEAM, OPP, DATE, GAME_ID, FGA, OREB, TOV, FTA, MIN.
        games: DataFrame with at least HOME, AWAY, DATE columns.
        windows: Rolling window sizes. Defaults to [5, 10, 20].

    Returns:
        Copy of games with added columns for each window W:
            HOME_PACE_L{W}  - home team's rolling average pace
            AWAY_PACE_L{W}  - away team's rolling average pace
            PROJ_PACE_L{W}  - projected game pace (average of both teams)

        NaN occurs when a team has no prior games in the window.
    """
    _validate_logs(team_game_logs)
    _validate_games(games)

    per_game = _compute_per_game_pace(team_game_logs)
    rolling  = _compute_rolling(per_game, windows)
    result   = _attach_to_games(rolling, games, windows)
    return _add_projected_pace(result, windows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_per_game_pace(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute pace for each team-game row."""
    df = logs.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    poss = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]

    # Use actual game minutes if available; fall back to 48 for regulation
    minutes = pd.to_numeric(df["MIN"], errors="coerce").fillna(_REGULATION_MINUTES)
    minutes = minutes.replace(0, _REGULATION_MINUTES)

    df["PACE"] = (poss / minutes) * 48.0

    return df[["TEAM", "DATE", "GAME_ID", "PACE"]]


def _compute_rolling(per_game: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling mean of per-game pace for each team, sorted chronologically."""
    parts = []
    for _, grp in per_game.groupby("TEAM"):
        grp = grp.sort_values("DATE").copy()
        for w in windows:
            grp[f"PACE_L{w}"] = grp["PACE"].rolling(window=w, min_periods=1).mean()
        parts.append(grp)
    return pd.concat(parts).reset_index(drop=True) if parts else per_game


def _attach_to_games(
    rolling: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """Attach home and away pace rolling stats via merge_asof."""
    games = games.copy()
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values("DATE").reset_index(drop=True)

    feature_cols = [f"PACE_L{w}" for w in windows]
    rolling = rolling.sort_values(["TEAM", "DATE"])

    for side, col in [("HOME", "HOME"), ("AWAY", "AWAY")]:
        side_rolling = rolling[["TEAM", "DATE"] + feature_cols].copy()
        side_rolling = side_rolling.rename(columns={"TEAM": col})
        rename_map = {f: f"{side}_{f}" for f in feature_cols}
        side_rolling = side_rolling.rename(columns=rename_map)

        games = pd.merge_asof(
            games,
            side_rolling.sort_values("DATE"),
            on="DATE",
            by=col,
            direction="backward",
        )

    return games


def _add_projected_pace(games: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add PROJ_PACE_L{W} = average of home and away pace."""
    for w in windows:
        h = f"HOME_PACE_L{w}"
        a = f"AWAY_PACE_L{w}"
        if h in games.columns and a in games.columns:
            games[f"PROJ_PACE_L{w}"] = (games[h] + games[a]) / 2.0
    return games


def _validate_logs(logs: pd.DataFrame) -> None:
    required = {"TEAM", "DATE", "GAME_ID", "FGA", "OREB", "TOV", "FTA", "MIN"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"team_game_logs missing columns: {sorted(missing)}")


def _validate_games(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "DATE"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games missing columns: {sorted(missing)}")

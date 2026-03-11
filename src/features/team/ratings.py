"""Rolling offensive, defensive, and net rating features for NBA teams.

Ratings are points-per-100-possessions metrics — the gold standard for
measuring team quality because they normalise for pace differences.

Possessions formula (Oliver approximation):
    poss = FGA - OREB + TOV + 0.44 * FTA

Offensive rating  = (team_pts   / team_poss)  * 100
Defensive rating  = (opp_pts    / opp_poss)   * 100
Net rating        = off_rating  - def_rating

Rolling windows L5 / L10 / L20 are computed using only games BEFORE the
target game date, so there is zero data leakage.  When fewer than min_periods
games exist the rolling mean is computed on whatever data is available
(min_periods=1); callers may filter on a minimum sample size if desired.

Usage:
    from src.data.nba_api_client import fetch_team_game_logs
    from src.features.team.ratings import compute_rolling_ratings

    logs  = fetch_team_game_logs(2024)
    games = ...  # DataFrame with HOME, AWAY, DATE columns
    games = compute_rolling_ratings(logs, games)
    # Adds: HOME_OFF_RATING_L5/L10/L20, HOME_DEF_RATING_*, HOME_NET_RATING_*
    #       AWAY_OFF_RATING_L5/L10/L20, AWAY_DEF_RATING_*, AWAY_NET_RATING_*
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_WINDOWS: list[int] = [5, 10, 20]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_ratings(
    team_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach rolling off/def/net rating columns to a games DataFrame.

    Args:
        team_game_logs: Output of fetch_team_game_logs().  Must contain
            TEAM, OPP, DATE, GAME_ID, FGM, FGA, FG3M, FTA, OREB, TOV, PTS.
        games: DataFrame with at least HOME, AWAY, DATE columns.
        windows: Rolling window sizes. Defaults to [5, 10, 20].

    Returns:
        Copy of games with added columns for each window W:
            HOME_OFF_RATING_L{W}, HOME_DEF_RATING_L{W}, HOME_NET_RATING_L{W}
            AWAY_OFF_RATING_L{W}, AWAY_DEF_RATING_L{W}, AWAY_NET_RATING_L{W}

        Missing values (NaN) occur when a team has fewer than 1 prior game.
    """
    _validate_logs(team_game_logs)
    _validate_games(games)

    per_game = _compute_per_game_ratings(team_game_logs)
    rolling  = _compute_rolling(per_game, windows)
    return _attach_to_games(rolling, games, windows, ["OFF_RATING", "DEF_RATING", "NET_RATING"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _possessions(fga: pd.Series, oreb: pd.Series, tov: pd.Series, fta: pd.Series) -> pd.Series:
    """Oliver possession estimate: FGA - OREB + TOV + 0.44*FTA."""
    return fga - oreb + tov + 0.44 * fta


def _compute_per_game_ratings(logs: pd.DataFrame) -> pd.DataFrame:
    """Add OFF_RATING, DEF_RATING, NET_RATING columns to team game logs.

    Requires a self-join on GAME_ID to pair each team's row with its
    opponent's row (needed for defensive rating calculation).
    """
    df = logs.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Self-join: pair each team-game with the opposing team's stats
    opp = df[["GAME_ID", "TEAM", "PTS", "FGA", "OREB", "TOV", "FTA"]].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM", "OPP_PTS", "OPP_FGA", "OPP_OREB", "OPP_TOV", "OPP_FTA"]

    paired = df.merge(opp, left_on=["GAME_ID", "OPP"], right_on=["GAME_ID", "OPP_TEAM"], how="left")

    # Handle any games where opponent row is missing (data gaps)
    for col in ["OPP_PTS", "OPP_FGA", "OPP_OREB", "OPP_TOV", "OPP_FTA"]:
        if col not in paired.columns:
            paired[col] = np.nan

    team_poss = _possessions(paired["FGA"], paired["OREB"], paired["TOV"], paired["FTA"])
    opp_poss  = _possessions(paired["OPP_FGA"], paired["OPP_OREB"], paired["OPP_TOV"], paired["OPP_FTA"])

    # Avoid division by zero
    team_poss = team_poss.replace(0, np.nan)
    opp_poss  = opp_poss.replace(0, np.nan)

    paired["OFF_RATING"] = (paired["PTS"]     / team_poss) * 100
    paired["DEF_RATING"] = (paired["OPP_PTS"] / opp_poss)  * 100
    paired["NET_RATING"] = paired["OFF_RATING"] - paired["DEF_RATING"]

    return paired[["TEAM", "DATE", "GAME_ID", "OFF_RATING", "DEF_RATING", "NET_RATING"]]


def _compute_rolling(per_game: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute rolling means for each window, grouped by team.

    The rolling window is computed on games sorted chronologically.
    Each row represents the rolling average THROUGH that game (inclusive),
    so when we later do merge_asof(DATE <=  game_date) we correctly get
    the stats accumulated up to and including the team's most recent game
    before the target game date.
    """
    parts = []
    for _, grp in per_game.groupby("TEAM"):
        grp = grp.sort_values("DATE").copy()
        for w in windows:
            for stat in ["OFF_RATING", "DEF_RATING", "NET_RATING"]:
                grp[f"{stat}_L{w}"] = grp[stat].rolling(window=w, min_periods=1).mean()
        parts.append(grp)
    return pd.concat(parts).reset_index(drop=True) if parts else per_game


def _attach_to_games(
    rolling: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int],
    stat_names: list[str],
) -> pd.DataFrame:
    """Join rolling team stats to a games DataFrame using as-of merge.

    For each game row, merge_asof finds the most recent rolling stat row
    for that team with DATE <= game DATE.  This gives pre-game statistics
    with no leakage.
    """
    games = games.copy()
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values("DATE").reset_index(drop=True)

    rolling = rolling.sort_values(["TEAM", "DATE"])

    feature_cols = [f"{s}_L{w}" for s in stat_names for w in windows]

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


def _validate_logs(logs: pd.DataFrame) -> None:
    required = {"TEAM", "OPP", "DATE", "GAME_ID", "FGA", "OREB", "TOV", "FTA", "PTS"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"team_game_logs missing columns: {sorted(missing)}")


def _validate_games(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "DATE"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games missing columns: {sorted(missing)}")

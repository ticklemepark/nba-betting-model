"""Dean Oliver's Four Factors rolling features for NBA teams.

The Four Factors are the best box-score proxies for team quality:

    1. Effective FG%  (eFG%)  = (FGM + 0.5 * FG3M) / FGA
       — Weights 3-pointers correctly (worth 50% more than 2-pointers).

    2. Turnover Rate  (TOV%)  = TOV / poss * 100
       — Share of possessions ending in turnovers.

    3. Offensive Rebound Rate (ORB%) = OREB / (OREB + opp_DREB)
       — Share of missed shots recovered for second chances.
       Requires opponent DREB via GAME_ID join.

    4. Free-Throw Rate  (FTR)  = FTA / FGA
       — Ability to get to the line.

Rolling windows L5 / L10 / L20 are pre-game (no leakage) via merge_asof.

Usage:
    from src.data.nba_api_client import fetch_team_game_logs
    from src.features.team.four_factors import compute_rolling_four_factors

    logs  = fetch_team_game_logs(2024)
    games = compute_rolling_four_factors(logs, games)
    # Adds: HOME_EFG_PCT_L*/TOV_PCT_L*/ORB_PCT_L*/FTR_L*
    #       AWAY_EFG_PCT_L*/TOV_PCT_L*/ORB_PCT_L*/FTR_L*
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_WINDOWS: list[int] = [5, 10, 20]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_four_factors(
    team_game_logs: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach rolling Four Factor columns to a games DataFrame.

    Args:
        team_game_logs: Output of fetch_team_game_logs(). Must contain
            TEAM, OPP, DATE, GAME_ID, FGM, FGA, FG3M, FTA, TOV, OREB, DREB.
        games: DataFrame with at least HOME, AWAY, DATE columns.
        windows: Rolling window sizes. Defaults to [5, 10, 20].

    Returns:
        Copy of games with added columns for each window W:
            HOME_EFG_PCT_L{W}  - Effective FG% (higher is better offensively)
            HOME_TOV_PCT_L{W}  - Turnover rate % (lower is better)
            HOME_ORB_PCT_L{W}  - Offensive rebound % (higher is better)
            HOME_FTR_L{W}      - Free throw rate FTA/FGA (higher is better)
            AWAY_*             - same for away team

        NaN occurs when a team has no prior games.
    """
    _validate_logs(team_game_logs)
    _validate_games(games)

    per_game = _compute_per_game_factors(team_game_logs)
    rolling  = _compute_rolling(per_game, windows)
    return _attach_to_games(rolling, games, windows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_per_game_factors(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game Four Factor values for each team-game row."""
    df = logs.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Possession estimate (needed for TOV%)
    poss = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    poss = poss.replace(0, np.nan)

    # Factor 1: Effective FG%
    df["EFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)

    # Factor 2: Turnover rate
    df["TOV_PCT"] = df["TOV"] / poss * 100

    # Factor 4: Free throw rate (no self-join needed)
    df["FTR"] = df["FTA"] / df["FGA"].replace(0, np.nan)

    # Factor 3: Offensive rebound % — needs opponent DREB (self-join)
    opp_dreb = (
        df[["GAME_ID", "TEAM", "DREB"]]
        .rename(columns={"TEAM": "OPP_TEAM", "DREB": "OPP_DREB"})
    )
    paired = df.merge(opp_dreb, left_on=["GAME_ID", "OPP"], right_on=["GAME_ID", "OPP_TEAM"], how="left")
    oreb_plus_opp_dreb = paired["OREB"] + paired["OPP_DREB"].fillna(0)
    oreb_plus_opp_dreb = oreb_plus_opp_dreb.replace(0, np.nan)
    df["ORB_PCT"] = paired["OREB"].values / oreb_plus_opp_dreb.values * 100

    return df[["TEAM", "DATE", "GAME_ID", "EFG_PCT", "TOV_PCT", "ORB_PCT", "FTR"]]


def _compute_rolling(per_game: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling mean of per-game Four Factors for each team."""
    stats = ["EFG_PCT", "TOV_PCT", "ORB_PCT", "FTR"]
    parts = []
    for _, grp in per_game.groupby("TEAM"):
        grp = grp.sort_values("DATE").copy()
        for w in windows:
            for stat in stats:
                grp[f"{stat}_L{w}"] = grp[stat].rolling(window=w, min_periods=1).mean()
        parts.append(grp)
    return pd.concat(parts).reset_index(drop=True) if parts else per_game


def _attach_to_games(
    rolling: pd.DataFrame,
    games: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """Attach home and away Four Factor rolling stats via merge_asof."""
    games = games.copy()
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values("DATE").reset_index(drop=True)

    stats = ["EFG_PCT", "TOV_PCT", "ORB_PCT", "FTR"]
    feature_cols = [f"{s}_L{w}" for s in stats for w in windows]
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


def _validate_logs(logs: pd.DataFrame) -> None:
    required = {"TEAM", "OPP", "DATE", "GAME_ID", "FGM", "FGA", "FG3M", "FTA", "TOV", "OREB", "DREB"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"team_game_logs missing columns: {sorted(missing)}")


def _validate_games(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "DATE"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games missing columns: {sorted(missing)}")

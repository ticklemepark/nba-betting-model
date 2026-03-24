"""Rolling usage rate features for NBA players.

Usage rate measures what fraction of a team's offensive possessions a player
uses while on the floor.  High usage → more shots, free throws, and turnovers,
which means higher stat ceilings for points and assists.

Because per-game possession counts for individual players aren't in the basic
game log, we use a per-minute volume proxy that is proportional to true usage:

    usage_proxy = (FGA + 0.44 * FTA + TOV) / MAX(MIN, 1) * 36

This normalises to per-36-minutes so the value is comparable across players
with different roles (starters vs. bench).  Rolling windows capture whether a
player is trending toward higher or lower usage (e.g. after a teammate's injury).

Usage:
    from src.features.player.usage import compute_rolling_usage

    player_games = compute_rolling_usage(player_log, player_games, windows=[5, 10, 20])
    # Adds: USAGE_PROXY_L5, USAGE_PROXY_L10, USAGE_PROXY_L20, USAGE_PROXY_SEASON
    # Also adds: MIN_L5, MIN_L10, MIN_L20, MIN_SEASON  (minutes context)
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_WINDOWS: list[int] = [5, 10, 20]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rolling_usage(
    player_game_logs: pd.DataFrame,
    player_games: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Attach rolling usage proxy and minutes columns to player_games.

    Args:
        player_game_logs: Output of fetch_player_game_logs(). Must contain
            PLAYER_ID, DATE, SEASON, FGA, FTA, TOV, MIN.
        player_games: One row per player per upcoming game. Must contain
            PLAYER_ID, DATE.
        windows: Rolling window sizes. Defaults to [5, 10, 20].

    Returns:
        Copy of player_games with added columns per window W:
            USAGE_PROXY_L{W}  - rolling mean of per-36-min usage proxy
            MIN_L{W}          - rolling mean of minutes played
            USAGE_PROXY_SEASON - season-to-date usage proxy mean
            MIN_SEASON         - season-to-date minutes mean

        NaN where a player has no prior games in window.
    """
    _validate_logs(player_game_logs)
    _validate_player_games(player_games)

    enriched = _compute_usage_proxy(player_game_logs)
    rolling  = _compute_rolling(enriched, windows)
    return _attach_to_player_games(rolling, player_games, windows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_usage_proxy(logs: pd.DataFrame) -> pd.DataFrame:
    """Add USAGE_PROXY column: (FGA + 0.44*FTA + TOV) / min * 36."""
    df = logs.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    safe_min = df["MIN"].clip(lower=1.0)
    df["USAGE_PROXY"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / safe_min * 36.0
    return df


def _compute_rolling(logs: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Per-player rolling means for USAGE_PROXY and MIN."""
    parts = []
    for _, grp in logs.groupby("PLAYER_ID"):
        grp = grp.sort_values("DATE").copy()
        for w in windows:
            grp[f"USAGE_PROXY_L{w}"] = (
                grp["USAGE_PROXY"].rolling(window=w, min_periods=1).mean()
            )
            grp[f"MIN_L{w}"] = grp["MIN"].rolling(window=w, min_periods=1).mean()

        for season_val, season_grp in grp.groupby("SEASON"):
            idx = season_grp.index
            grp.loc[idx, "USAGE_PROXY_SEASON"] = (
                season_grp["USAGE_PROXY"].expanding(min_periods=1).mean().values
            )
            grp.loc[idx, "MIN_SEASON"] = (
                season_grp["MIN"].expanding(min_periods=1).mean().values
            )
        parts.append(grp)

    return pd.concat(parts).reset_index(drop=True) if parts else logs


def _attach_to_player_games(
    rolling: pd.DataFrame,
    player_games: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    player_games = player_games.copy()
    player_games["DATE"] = pd.to_datetime(player_games["DATE"])
    player_games = player_games.sort_values("DATE").reset_index(drop=True)

    feature_cols = (
        [f"USAGE_PROXY_L{w}" for w in windows]
        + [f"MIN_L{w}" for w in windows]
        + ["USAGE_PROXY_SEASON", "MIN_SEASON"]
    )
    available = [c for c in feature_cols if c in rolling.columns]
    rolling_slim = rolling[["PLAYER_ID", "DATE"] + available].sort_values("DATE")

    player_games["PLAYER_ID"] = player_games["PLAYER_ID"].astype(object)
    rolling_slim["PLAYER_ID"] = rolling_slim["PLAYER_ID"].astype(object)
    return pd.merge_asof(
        player_games,
        rolling_slim,
        on="DATE",
        by="PLAYER_ID",
        direction="backward",
    )


def _validate_logs(logs: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE", "SEASON", "FGA", "FTA", "TOV", "MIN"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"player_game_logs missing columns: {sorted(missing)}")


def _validate_player_games(player_games: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")

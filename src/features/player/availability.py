"""Teammate absence usage redistribution features.

When a high-usage player is ruled out, the remaining players absorb extra
possessions.  This is one of the highest-confidence edges on Underdog:
lines are slow to adjust for late injury news.

Model:
    absent_volume = absent_player's rolling average (FGA + 0.44*FTA + TOV)
    team_remaining_volume = sum of active players' rolling usage proxies

    For each active player on the same team:
        teammate_out_boost = absent_volume * (player_usage / team_remaining_volume)

This produces a per-player estimated *additional* offensive volume they absorb,
which can be added to their baseline projection.

Usage:
    from src.features.player.availability import compute_teammate_out_boost

    # absent_players: list of dicts {player_id, team, date, season}
    player_games = compute_teammate_out_boost(
        player_log, player_games, absent_players
    )
    # Adds: TEAMMATE_OUT_BOOST (estimated extra FGA+equiv. possessions per game)
    #       TEAMMATE_OUT_FLAG  (1 if any teammate is out, 0 otherwise)
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Rolling window to use for estimating absent player's usual volume
_USAGE_WINDOW = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_teammate_out_boost(
    player_game_logs: pd.DataFrame,
    player_games: pd.DataFrame,
    absent_players: list[dict],
) -> pd.DataFrame:
    """Attach teammate-absence usage boost to player_games.

    Args:
        player_game_logs: Output of fetch_player_game_logs(). Must contain
            PLAYER_ID, DATE, TEAM, FGA, FTA, TOV, MIN.
        player_games: One row per active player per upcoming game. Must contain
            PLAYER_ID, TEAM, DATE.
        absent_players: List of dicts, each with keys:
            player_id - the absent player's ID
            team      - team abbreviation (3-letter)
            date      - game date (YYYY-MM-DD string or datetime)

    Returns:
        Copy of player_games with added columns:
            TEAMMATE_OUT_BOOST - estimated extra possession-volume absorbed
            TEAMMATE_OUT_FLAG  - 1 if at least one teammate is out

        A boost of 3.5 means the player is expected to absorb ~3.5 extra
        "possession-equivalent" opportunities (FGA + 0.44*FTA + TOV units).
    """
    _validate_logs(player_game_logs)
    _validate_player_games(player_games)

    if not absent_players:
        result = player_games.copy()
        result["TEAMMATE_OUT_BOOST"] = 0.0
        result["TEAMMATE_OUT_FLAG"] = 0
        return result

    usage_rolling = _compute_usage_rolling(player_game_logs)
    return _apply_boost(usage_rolling, player_games, absent_players)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _volume(fga: pd.Series, fta: pd.Series, tov: pd.Series) -> pd.Series:
    """Offensive possession volume: FGA + 0.44*FTA + TOV."""
    return fga + 0.44 * fta + tov


def _compute_usage_rolling(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling usage volume (L10) for every player."""
    logs = logs.copy()
    logs["DATE"] = pd.to_datetime(logs["DATE"])
    logs["VOLUME"] = _volume(logs["FGA"], logs["FTA"], logs["TOV"])

    parts = []
    for _, grp in logs.groupby("PLAYER_ID"):
        grp = grp.sort_values("DATE").copy()
        grp[f"VOLUME_L{_USAGE_WINDOW}"] = (
            grp["VOLUME"].rolling(window=_USAGE_WINDOW, min_periods=1).mean()
        )
        parts.append(grp)

    return pd.concat(parts).reset_index(drop=True) if parts else logs


def _apply_boost(
    usage_rolling: pd.DataFrame,
    player_games: pd.DataFrame,
    absent_players: list[dict],
) -> pd.DataFrame:
    """For each player-game, compute how much boost they receive from absences."""
    player_games = player_games.copy()
    player_games["DATE"] = pd.to_datetime(player_games["DATE"])

    boost_col = [0.0] * len(player_games)
    flag_col  = [0]  * len(player_games)

    # Index usage_rolling by (player_id, date) for fast lookup
    usage_rolling = usage_rolling.sort_values("DATE")

    def _get_usage(player_id, as_of_date: pd.Timestamp) -> float:
        """Most recent rolling usage for player_id before as_of_date."""
        rows = usage_rolling[
            (usage_rolling["PLAYER_ID"] == player_id) &
            (usage_rolling["DATE"] < as_of_date)
        ]
        if rows.empty:
            return 0.0
        return float(rows.iloc[-1][f"VOLUME_L{_USAGE_WINDOW}"])

    # Group absent players by (team, date)
    absent_by_team_date: dict[tuple, list] = {}
    for ap in absent_players:
        key = (ap["team"], pd.Timestamp(ap["date"]))
        absent_by_team_date.setdefault(key, []).append(ap["player_id"])

    for i, row in player_games.iterrows():
        team = row["TEAM"]
        date = row["DATE"]
        pid  = row["PLAYER_ID"]
        key  = (team, date)

        absent_ids = absent_by_team_date.get(key, [])
        if not absent_ids:
            continue

        # Total volume being freed up by absent teammates
        absent_volume = sum(_get_usage(aid, date) for aid in absent_ids)
        if absent_volume <= 0:
            continue

        # Active players on the same team in this game
        active_on_team = player_games[
            (player_games["TEAM"] == team) &
            (player_games["DATE"] == date) &
            (~player_games["PLAYER_ID"].isin(absent_ids))
        ]

        # Each active player's usage as a fraction of total active usage
        active_volumes = {
            r["PLAYER_ID"]: _get_usage(r["PLAYER_ID"], date)
            for _, r in active_on_team.iterrows()
        }
        total_active_volume = sum(active_volumes.values())

        if total_active_volume <= 0:
            continue

        player_share = active_volumes.get(pid, 0.0) / total_active_volume
        boost_col[i] = absent_volume * player_share
        flag_col[i]  = 1

    player_games["TEAMMATE_OUT_BOOST"] = boost_col
    player_games["TEAMMATE_OUT_FLAG"]  = flag_col
    return player_games


def _validate_logs(logs: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE", "TEAM", "FGA", "FTA", "TOV"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"player_game_logs missing columns: {sorted(missing)}")


def _validate_player_games(player_games: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "TEAM", "DATE"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")

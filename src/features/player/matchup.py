"""Player vs. specific opponent historical stat features.

Some players consistently outperform or underperform against specific teams —
a real matchup edge that Underdog's lines may not fully price in.  These
averages are computed season-to-date (all prior games vs. the same opponent
within the current season) to balance recency with sample size.

Usage:
    from src.features.player.matchup import compute_vs_opponent

    player_games = compute_vs_opponent(
        player_log, player_games,
        stats=["PTS", "REB", "AST"],
    )
    # Adds: PTS_VS_OPP_AVG, REB_VS_OPP_AVG, AST_VS_OPP_AVG
    #       PTS_VS_OPP_N (number of prior games vs. this opponent)
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_STATS: list[str] = ["PTS", "REB", "AST", "FG3M", "MIN"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_vs_opponent(
    player_game_logs: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str] = DEFAULT_STATS,
) -> pd.DataFrame:
    """Attach season-to-date vs-opponent averages to player_games.

    For each player-game, computes the player's mean stats against the
    specific opponent using only games played BEFORE the target date.

    Args:
        player_game_logs: Output of fetch_player_game_logs(). Must contain
            PLAYER_ID, DATE, SEASON, OPP, and all stats.
        player_games: One row per player per game. Must contain
            PLAYER_ID, DATE, SEASON, OPP.
        stats: Stat columns to compute matchup averages for.

    Returns:
        Copy of player_games with added columns per stat:
            {STAT}_VS_OPP_AVG - mean stat vs. this opponent this season
            {STAT}_VS_OPP_N   - number of prior games vs. this opponent

        NaN / 0 when player has never faced this opponent this season.
    """
    _validate_logs(player_game_logs, stats)
    _validate_player_games(player_games)

    return _compute_and_attach(player_game_logs, player_games, stats)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_and_attach(
    logs: pd.DataFrame,
    player_games: pd.DataFrame,
    stats: list[str],
) -> pd.DataFrame:
    """For each player-game row, look up prior stats vs the same opponent.

    This is a row-by-row operation keyed on (PLAYER_ID, OPP, SEASON).
    We build a lookup dict: (player_id, opp, season) -> {stat: [values]}
    then iterate player_games chronologically.
    """
    logs = logs.copy()
    logs["DATE"] = pd.to_datetime(logs["DATE"])
    player_games = player_games.copy()
    player_games["DATE"] = pd.to_datetime(player_games["DATE"])
    player_games = player_games.sort_values("DATE").reset_index(drop=True)

    # Sort logs chronologically — we'll process events in order
    logs = logs.sort_values("DATE").reset_index(drop=True)

    # State: (player_id, opp, season) -> list of per-game stat values
    history: dict[tuple, dict[str, list]] = {}

    # Pre-build history up to each date using a pointer approach
    # Combine logs and player_games into one timeline, sorted by date
    # For each player_game row, use only log rows with date < player_game date

    # Efficient approach: for each player_game row, binary-search logs
    logs_by_key: dict[tuple, pd.DataFrame] = {}
    for (pid, opp, season), grp in logs.groupby(["PLAYER_ID", "OPP", "SEASON"]):
        logs_by_key[(pid, opp, season)] = grp.sort_values("DATE")

    stat_avgs  = {s: [] for s in stats}
    stat_counts = []

    for _, row in player_games.iterrows():
        pid    = row["PLAYER_ID"]
        opp    = row["OPP"]
        season = row["SEASON"]
        date   = row["DATE"]

        key = (pid, opp, season)
        prior = logs_by_key.get(key)

        if prior is None or prior.empty:
            for s in stats:
                stat_avgs[s].append(float("nan"))
            stat_counts.append(0)
            continue

        # Games strictly before this date
        prior_before = prior[prior["DATE"] < date]

        if prior_before.empty:
            for s in stats:
                stat_avgs[s].append(float("nan"))
            stat_counts.append(0)
        else:
            for s in stats:
                stat_avgs[s].append(
                    prior_before[s].mean() if s in prior_before.columns else float("nan")
                )
            stat_counts.append(len(prior_before))

    result = player_games.copy()
    for s in stats:
        result[f"{s}_VS_OPP_AVG"] = stat_avgs[s]
    result["VS_OPP_N"] = stat_counts
    return result


def _validate_logs(logs: pd.DataFrame, stats: list[str]) -> None:
    required = {"PLAYER_ID", "DATE", "SEASON", "OPP"}
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"player_game_logs missing columns: {sorted(missing)}")


def _validate_player_games(player_games: pd.DataFrame) -> None:
    required = {"PLAYER_ID", "DATE", "SEASON", "OPP"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")

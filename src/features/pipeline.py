"""Feature pipeline orchestrator.

Ties all team-level and player-level feature modules together into two
top-level functions:

    build_game_features(games, team_logs, windows)
        → Full feature matrix for the game outcome model.
        → One row per game with ~40 pre-game features.

    build_player_features(player_games, player_logs, team_logs, absent_players, windows)
        → Full feature matrix for the player prop model.
        → One row per player per game with ~30 pre-game features.

Both functions are zero-leakage: every feature uses only data available
before the game tip-off time.

Usage:
    from src.data.nba_api_client import fetch_team_game_logs, fetch_player_game_logs
    from src.features.pipeline import build_game_features, build_player_features

    team_logs   = fetch_team_game_logs(2024)
    player_logs = fetch_player_game_logs(2024)

    # For game outcome model
    game_features = build_game_features(games, team_logs)

    # For player prop model
    player_features = build_player_features(player_games, player_logs, team_logs)
"""

import logging

import pandas as pd

from src.features.team.elo      import compute_elo_ratings
from src.features.team.form     import compute_streaks, compute_wins_rolling
from src.features.team.h2h      import compute_h2h_records
from src.features.team.schedule import compute_b2b_flags, compute_rest_days
from src.features.team.ratings  import compute_rolling_ratings
from src.features.team.pace     import compute_rolling_pace
from src.features.team.four_factors import compute_rolling_four_factors

from src.features.player.rolling_stats import compute_rolling_player_stats
from src.features.player.usage         import compute_rolling_usage
from src.features.player.home_away     import compute_home_away_splits
from src.features.player.matchup       import compute_vs_opponent
from src.features.player.availability  import compute_teammate_out_boost

log = logging.getLogger(__name__)

_DEFAULT_WINDOWS      = [5, 10, 20]
_DEFAULT_PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_game_features(
    games: pd.DataFrame,
    team_logs: pd.DataFrame,
    windows: list[int] = _DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Compute full game-level feature matrix.

    Runs all team-level feature modules in sequence.  Each module adds new
    columns to the games DataFrame without dropping existing ones.

    Args:
        games: Must contain HOME, AWAY, HOME_PTS, AWAY_PTS, DATE, SEASON.
        team_logs: Output of fetch_team_game_logs().
        windows: Rolling window sizes for nba_api-derived features.

    Returns:
        Input games DataFrame enriched with all team-level features:
        ELO, streaks, B2B, rest days, H2H, rolling ratings, pace,
        Four Factors, wins_L{W}.
    """
    log.info("Building game features for %d games...", len(games))

    # Phase 1 features (from existing modules)
    games = compute_elo_ratings(games)
    games = compute_streaks(games)
    games = compute_b2b_flags(games)
    games = compute_rest_days(games)
    games = compute_h2h_records(games)

    # Phase 2 features (nba_api-derived rolling stats)
    if not team_logs.empty:
        games = compute_rolling_ratings(team_logs, games, windows)
        games = compute_rolling_pace(team_logs, games, windows)
        games = compute_rolling_four_factors(team_logs, games, windows)
        games = compute_wins_rolling(games)

    log.info("Game feature matrix shape: %s", games.shape)
    return games


def build_player_features(
    player_games: pd.DataFrame,
    player_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    absent_players: list[dict] | None = None,
    stats: list[str] = _DEFAULT_PLAYER_STATS,
    windows: list[int] = _DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Compute full player-level feature matrix.

    Args:
        player_games: One row per player per upcoming game. Must contain
            PLAYER_ID, PLAYER_NAME, TEAM, OPP, DATE, SEASON, IS_HOME.
        player_logs: Output of fetch_player_game_logs().
        team_logs: Output of fetch_team_game_logs() (for pace context).
        absent_players: List of dicts {player_id, team, date} for today's DNPs.
        stats: Player stat columns to compute rolling averages for.
        windows: Rolling window sizes.

    Returns:
        Input player_games enriched with rolling stats, usage, home/away
        splits, matchup history, and teammate-absence boost.
    """
    log.info("Building player features for %d player-games...", len(player_games))

    if player_logs.empty:
        log.warning("player_logs is empty — returning player_games without features.")
        return player_games

    player_games = compute_rolling_player_stats(player_logs, player_games, stats, windows)
    player_games = compute_rolling_usage(player_logs, player_games, windows)
    player_games = compute_home_away_splits(player_logs, player_games, stats)

    if "OPP" in player_games.columns and "SEASON" in player_games.columns:
        player_games = compute_vs_opponent(player_logs, player_games, stats)

    if absent_players:
        player_games = compute_teammate_out_boost(player_logs, player_games, absent_players)
    else:
        player_games["TEAMMATE_OUT_BOOST"] = 0.0
        player_games["TEAMMATE_OUT_FLAG"]  = 0

    # Attach team pace context (high pace → more counting stats)
    if not team_logs.empty and "DATE" in player_games.columns:
        pace_cols = _get_pace_columns(player_games)
        if not pace_cols:
            pace_games = _stub_games_from_player_games(player_games)
            pace_games = compute_rolling_pace(team_logs, pace_games, windows)
            player_games = _attach_team_pace_to_players(player_games, pace_games, windows)

    log.info("Player feature matrix shape: %s", player_games.shape)
    return player_games


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pace_columns(player_games: pd.DataFrame) -> list[str]:
    """Return any PROJ_PACE columns already present."""
    return [c for c in player_games.columns if c.startswith("PROJ_PACE")]


def _stub_games_from_player_games(player_games: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal games DataFrame from player_games for pace computation."""
    cols = ["HOME", "AWAY", "DATE", "SEASON"]
    if not all(c in player_games.columns for c in ["TEAM", "OPP", "DATE"]):
        return pd.DataFrame(columns=cols)

    home_players = player_games[player_games.get("IS_HOME", pd.Series(True, index=player_games.index)).astype(bool)]
    if home_players.empty:
        return pd.DataFrame(columns=cols)

    stub = home_players.drop_duplicates(subset=["DATE", "TEAM"])[["TEAM", "OPP", "DATE"]].copy()
    stub = stub.rename(columns={"TEAM": "HOME", "OPP": "AWAY"})
    if "SEASON" in player_games.columns:
        stub = stub.merge(
            player_games[["DATE", "SEASON"]].drop_duplicates("DATE"),
            on="DATE", how="left"
        )
    return stub


def _attach_team_pace_to_players(
    player_games: pd.DataFrame,
    pace_games: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """Copy PROJ_PACE columns from pace_games to player_games via DATE+TEAM join."""
    pace_cols = [f"PROJ_PACE_L{w}" for w in windows if f"PROJ_PACE_L{w}" in pace_games.columns]
    if not pace_cols or "HOME" not in pace_games.columns:
        return player_games

    # Attach via home team (each game has one home team row in pace_games)
    home_pace = pace_games[["HOME", "DATE"] + pace_cols].rename(columns={"HOME": "TEAM"})
    away_pace = pace_games[["AWAY", "DATE"] + pace_cols].rename(columns={"AWAY": "TEAM"})
    team_pace = pd.concat([home_pace, away_pace]).drop_duplicates(subset=["TEAM", "DATE"])

    player_games["DATE"] = pd.to_datetime(player_games["DATE"])
    team_pace["DATE"] = pd.to_datetime(team_pace["DATE"])

    return player_games.merge(team_pace, on=["TEAM", "DATE"], how="left")

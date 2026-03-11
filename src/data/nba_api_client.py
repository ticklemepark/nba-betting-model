"""nba_api wrapper for fetching per-game team and player statistics.

All public functions return clean DataFrames using nba_api abbreviations
(BKN, CHA, PHX — NOT the bbref variants BRK, CHO, PHO).  Callers that
join with bbref data must apply the BBREF_TO_NBA_API map from constants.

Rate limiting: 1-second sleep after every API call to avoid throttling.

Usage:
    from src.data.nba_api_client import fetch_team_game_logs, fetch_player_game_logs

    team_logs  = fetch_team_game_logs(2024)   # 2023-24 regular season
    player_logs = fetch_player_game_logs(2024)
"""

import logging
import time

import pandas as pd
from nba_api.stats.endpoints import teamgamelogs, playergamelogs

log = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 1.0
_SEASON_TYPE = "Regular Season"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_team_game_logs(season: int) -> pd.DataFrame:
    """Fetch per-game box scores for all teams in a regular season.

    Args:
        season: Season END year (e.g. 2024 for the 2023-24 season).

    Returns:
        DataFrame with one row per team per game, sorted by DATE.
        Columns:
            TEAM       - NBA API team abbreviation (e.g. BKN, CHA, PHX)
            OPP        - Opponent abbreviation
            DATE       - Game date (YYYY-MM-DD string)
            SEASON     - Season end year (int)
            IS_HOME    - True if team played at home
            WL         - 'W' or 'L'
            GAME_ID    - nba_api game identifier
            MIN        - Game duration in minutes (48 for regulation, 53 for 1OT, etc.)
            FGM, FGA, FG3M, FG3A, FTM, FTA
            OREB, DREB, REB, AST, TOV, STL, BLK, PF, PTS, PLUS_MINUS
    """
    season_str = _season_str(season)
    log.info("Fetching team game logs for %s...", season_str)

    try:
        endpoint = teamgamelogs.TeamGameLogs(
            season_nullable=season_str,
            season_type_nullable=_SEASON_TYPE,
            timeout=30,
        )
        time.sleep(_RATE_LIMIT_SECONDS)
        raw = endpoint.get_data_frames()[0]
    except Exception as exc:
        log.error("Failed to fetch team game logs for %s: %s", season_str, exc)
        return pd.DataFrame(columns=_TEAM_LOG_COLUMNS)

    if raw.empty:
        log.warning("No team game log data returned for %s", season_str)
        return pd.DataFrame(columns=_TEAM_LOG_COLUMNS)

    return _clean_team_game_logs(raw, season)


def fetch_player_game_logs(season: int) -> pd.DataFrame:
    """Fetch per-game box scores for all players in a regular season.

    Args:
        season: Season END year (e.g. 2024 for the 2023-24 season).

    Returns:
        DataFrame with one row per player per game, sorted by DATE.
        Columns:
            PLAYER_ID, PLAYER_NAME
            TEAM       - NBA API team abbreviation
            OPP        - Opponent abbreviation
            DATE       - Game date (YYYY-MM-DD string)
            SEASON     - Season end year (int)
            IS_HOME    - True if player's team played at home
            WL         - 'W' or 'L'
            GAME_ID
            MIN        - Float minutes played
            FGM, FGA, FG3M, FG3A, FTM, FTA
            OREB, DREB, REB, AST, TOV, STL, BLK, PF, PTS, PLUS_MINUS
    """
    season_str = _season_str(season)
    log.info("Fetching player game logs for %s...", season_str)

    try:
        endpoint = playergamelogs.PlayerGameLogs(
            season_nullable=season_str,
            season_type_nullable=_SEASON_TYPE,
            timeout=30,
        )
        time.sleep(_RATE_LIMIT_SECONDS)
        raw = endpoint.get_data_frames()[0]
    except Exception as exc:
        log.error("Failed to fetch player game logs for %s: %s", season_str, exc)
        return pd.DataFrame(columns=_PLAYER_LOG_COLUMNS)

    if raw.empty:
        log.warning("No player game log data returned for %s", season_str)
        return pd.DataFrame(columns=_PLAYER_LOG_COLUMNS)

    return _clean_player_game_logs(raw, season)


# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_TEAM_LOG_COLUMNS: list[str] = [
    "TEAM", "OPP", "DATE", "SEASON", "IS_HOME", "WL", "GAME_ID",
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS",
    "PLUS_MINUS",
]

_PLAYER_LOG_COLUMNS: list[str] = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM", "OPP", "DATE", "SEASON",
    "IS_HOME", "WL", "GAME_ID",
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS",
    "PLUS_MINUS",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _season_str(season: int) -> str:
    """Convert season end year to nba_api format: 2024 -> '2023-24'."""
    return f"{season - 1}-{str(season)[2:]}"


def _parse_opponent(matchup: str) -> str:
    """Extract opponent abbreviation from MATCHUP string.

    'LAL vs. BOS' -> 'BOS'  (home game)
    'LAL @ BOS'   -> 'BOS'  (away game)
    """
    if " vs. " in matchup:
        return matchup.split(" vs. ")[1].strip()
    if " @ " in matchup:
        return matchup.split(" @ ")[1].strip()
    return ""


def _parse_min(min_val) -> float:
    """Parse MIN field to float minutes.

    nba_api may return '48:00' or 48 or '48'.
    """
    if pd.isna(min_val):
        return 0.0
    s = str(min_val).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _clean_team_game_logs(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Normalize raw TeamGameLogs response into our clean schema."""
    df = raw.copy()

    df["TEAM"]    = df["TEAM_ABBREVIATION"]
    df["OPP"]     = df["MATCHUP"].apply(_parse_opponent)
    df["DATE"]    = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    df["SEASON"]  = season
    df["IS_HOME"] = df["MATCHUP"].str.contains(r"vs\.", regex=True)
    df["WL"]      = df["WL"].str.strip()
    df["MIN"]     = df["MIN"].apply(_parse_min)

    # Rename stat columns (nba_api uses FG3M for 3-pointers)
    rename = {
        "FG3M": "FG3M", "FG3A": "FG3A",  # already correct
        "REB": "REB",
    }
    # Drop rank columns and anything we don't need
    keep = [
        "TEAM", "OPP", "DATE", "SEASON", "IS_HOME", "WL", "GAME_ID",
        "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS",
        "PLUS_MINUS",
    ]
    # Some columns might be missing in older API versions
    available = [c for c in keep if c in df.columns]
    missing_cols = [c for c in keep if c not in df.columns]
    if missing_cols:
        log.warning("Team game logs missing columns: %s", missing_cols)

    result = df[available].copy()

    # Coerce numeric columns
    int_cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS"]
    for col in int_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)

    if "PLUS_MINUS" in result.columns:
        result["PLUS_MINUS"] = pd.to_numeric(result["PLUS_MINUS"], errors="coerce").fillna(0).astype(int)

    return result.sort_values("DATE").reset_index(drop=True)


def _clean_player_game_logs(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Normalize raw PlayerGameLogs response into our clean schema."""
    df = raw.copy()

    df["PLAYER_ID"]   = df["PLAYER_ID"]
    df["PLAYER_NAME"] = df["PLAYER_NAME"]
    df["TEAM"]        = df["TEAM_ABBREVIATION"]
    df["OPP"]         = df["MATCHUP"].apply(_parse_opponent)
    df["DATE"]        = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    df["SEASON"]      = season
    df["IS_HOME"]     = df["MATCHUP"].str.contains(r"vs\.", regex=True)
    df["WL"]          = df["WL"].str.strip()
    df["MIN"]         = df["MIN"].apply(_parse_min)

    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM", "OPP", "DATE", "SEASON",
        "IS_HOME", "WL", "GAME_ID",
        "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS",
        "PLUS_MINUS",
    ]
    available = [c for c in keep if c in df.columns]
    missing_cols = [c for c in keep if c not in df.columns]
    if missing_cols:
        log.warning("Player game logs missing columns: %s", missing_cols)

    result = df[available].copy()

    # Coerce numeric columns
    int_cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS"]
    for col in int_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)

    if "PLUS_MINUS" in result.columns:
        result["PLUS_MINUS"] = pd.to_numeric(result["PLUS_MINUS"], errors="coerce").fillna(0).astype(int)

    result["MIN"] = pd.to_numeric(result["MIN"], errors="coerce").fillna(0.0)

    return result.sort_values("DATE").reset_index(drop=True)

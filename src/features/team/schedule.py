"""Schedule-based features for NBA teams.

Refactored from notebooks/feature-engineering.ipynb plus Phase 2 additions.

Functions:
    compute_b2b_flags   — Binary back-to-back flags (B2B = played yesterday).
    compute_rest_days   — Continuous rest days since last game (1 = B2B, 7+ capped).

A team is on a back-to-back if they played a game the previous calendar day.
Month and year boundaries are handled correctly by using timedelta arithmetic
rather than string-based date comparisons.
"""

import pandas as pd

_REST_CAP = 7   # Days of rest beyond this are treated identically


def compute_b2b_flags(games: pd.DataFrame) -> pd.DataFrame:
    """Compute back-to-back flags for home and away teams entering each game.

    Args:
        games: DataFrame with columns HOME, AWAY, DATE.
               DATE must be parseable as datetime (str YYYY-MM-DD or datetime).

    Returns:
        Input DataFrame sorted by DATE with added columns:
            HOME_B2B - 1 if home team played the previous calendar day, else 0
            AWAY_B2B - 1 if away team played the previous calendar day, else 0

        Safe to use as model features (pre-game only, no leakage).
    """
    _validate_columns(games)

    games = games.copy()
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values("DATE").reset_index(drop=True)

    last_game_date: dict[str, pd.Timestamp] = {}  # team -> date of most recent game

    home_b2b: list[int] = []
    away_b2b: list[int] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        date = row["DATE"]

        home_b2b.append(_is_b2b(home, date, last_game_date))
        away_b2b.append(_is_b2b(away, date, last_game_date))

        last_game_date[home] = date
        last_game_date[away] = date

    result = games.copy()
    result["HOME_B2B"] = home_b2b
    result["AWAY_B2B"] = away_b2b
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_b2b(team: str, date: pd.Timestamp, last_game_date: dict) -> int:
    """Return 1 if team played exactly one calendar day before `date`, else 0."""
    if team not in last_game_date:
        return 0
    return 1 if (date - last_game_date[team]).days == 1 else 0


def compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Compute days of rest since each team's previous game.

    Args:
        games: DataFrame with columns HOME, AWAY, DATE.
               DATE must be parseable as datetime (str YYYY-MM-DD or datetime).

    Returns:
        Input DataFrame sorted by DATE with added columns:
            HOME_REST  - days since home team's last game (capped at 7).
                         7 means 7+ days rest (season opener or long break).
            AWAY_REST  - days since away team's last game (capped at 7).

        A value of 1 is equivalent to a back-to-back (played yesterday).
        A value of 7 indicates a long rest or season start.

        Safe to use as model features (pre-game only, no leakage).
    """
    _validate_columns(games)

    games = games.copy()
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values("DATE").reset_index(drop=True)

    last_game_date: dict[str, pd.Timestamp] = {}

    home_rest: list[int] = []
    away_rest: list[int] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        date = row["DATE"]

        home_rest.append(_days_rest(home, date, last_game_date))
        away_rest.append(_days_rest(away, date, last_game_date))

        last_game_date[home] = date
        last_game_date[away] = date

    result = games.copy()
    result["HOME_REST"] = home_rest
    result["AWAY_REST"] = away_rest
    return result


def _days_rest(team: str, date: pd.Timestamp, last_game_date: dict) -> int:
    """Return days since team's last game, capped at _REST_CAP.

    Returns _REST_CAP if team has never played (season opener).
    """
    if team not in last_game_date:
        return _REST_CAP
    days = (date - last_game_date[team]).days
    return min(days, _REST_CAP)


def _validate_columns(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "DATE"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(
            f"games DataFrame is missing required columns: {sorted(missing)}"
        )

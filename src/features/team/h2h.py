"""Head-to-head record tracker for NBA teams.

Refactored from notebooks/feature-engineering.ipynb.

Tracks within-season win rates between specific pairs of teams.
Records reset at the start of each new season.
"""

import pandas as pd


def compute_h2h_records(games: pd.DataFrame) -> pd.DataFrame:
    """Compute within-season head-to-head win rates entering each game.

    Args:
        games: DataFrame with columns HOME, AWAY, HOME_PTS, AWAY_PTS,
               DATE, SEASON.

    Returns:
        Input DataFrame sorted by DATE with added columns:
            HOME_REC - home team's win rate vs this away team this season
            AWAY_REC - away team's win rate vs this home team this season

        Both are 0.0 when the teams have not met this season yet.
        When prior meetings exist: HOME_REC + AWAY_REC == 1.0.
        Safe to use as model features (pre-game only, no leakage).
    """
    _validate_columns(games)

    games = games.sort_values("DATE").reset_index(drop=True)

    # (team, opponent, season) -> [wins, total_games]
    h2h: dict[tuple, list[int]] = {}

    home_recs: list[float] = []
    away_recs: list[float] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        season = row["SEASON"]

        home_recs.append(_get_record(home, away, season, h2h))
        away_recs.append(_get_record(away, home, season, h2h))

        _update_h2h(home, away, season, home_won=row["HOME_PTS"] > row["AWAY_PTS"], h2h=h2h)

    result = games.copy()
    result["HOME_REC"] = home_recs
    result["AWAY_REC"] = away_recs
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_record(
    team: str,
    opp: str,
    season: int,
    h2h: dict,
) -> float:
    """Return team's win rate vs opp this season. 0.0 if no prior meetings."""
    key = (team, opp, season)
    if key not in h2h or h2h[key][1] == 0:
        return 0.0
    wins, total = h2h[key]
    return wins / total


def _update_h2h(
    home: str,
    away: str,
    season: int,
    home_won: bool,
    h2h: dict,
) -> None:
    """Record game result into both teams' h2h entries."""
    home_key = (home, away, season)
    away_key = (away, home, season)

    if home_key not in h2h:
        h2h[home_key] = [0, 0]
    if away_key not in h2h:
        h2h[away_key] = [0, 0]

    if home_won:
        h2h[home_key][0] += 1
    else:
        h2h[away_key][0] += 1

    h2h[home_key][1] += 1
    h2h[away_key][1] += 1


def _validate_columns(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(
            f"games DataFrame is missing required columns: {sorted(missing)}"
        )

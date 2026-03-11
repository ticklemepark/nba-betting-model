"""Win streak and recent form features for NBA teams.

Refactored from notebooks/feature-engineering.ipynb plus Phase 2 additions.

Functions:
    compute_streaks    — Consecutive wins entering each game (resets on loss/season).
    compute_wins_rolling — Wins in the last N games (rolling count, default window=10).
"""

import pandas as pd


def compute_streaks(games: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive win streaks for home and away teams entering each game.

    Args:
        games: DataFrame with columns HOME, AWAY, HOME_PTS, AWAY_PTS,
               DATE, SEASON.

    Returns:
        Input DataFrame sorted by DATE with added columns:
            HOME_STREAK - consecutive wins entering the game for the home team
            AWAY_STREAK - consecutive wins entering the game for the away team

        Both values are 0 at the start of each new season and after any loss.
        Safe to use as model features (pre-game only, no leakage).
    """
    _validate_columns(games)

    games = games.sort_values("DATE").reset_index(drop=True)

    streak: dict[str, int] = {}       # team -> current win streak
    season_seen: dict[str, int] = {}  # team -> last season seen

    home_streaks: list[int] = []
    away_streaks: list[int] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        season = row["SEASON"]

        home_streaks.append(_get_pre_game_streak(home, season, streak, season_seen))
        away_streaks.append(_get_pre_game_streak(away, season, streak, season_seen))

        home_won = row["HOME_PTS"] > row["AWAY_PTS"]
        streak[home] = (streak.get(home, 0) + 1) if home_won else 0
        streak[away] = 0 if home_won else (streak.get(away, 0) + 1)
        season_seen[home] = season
        season_seen[away] = season

    result = games.copy()
    result["HOME_STREAK"] = home_streaks
    result["AWAY_STREAK"] = away_streaks
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pre_game_streak(
    team: str,
    season: int,
    streak: dict[str, int],
    season_seen: dict[str, int],
) -> int:
    """Return pre-game win streak, resetting to 0 at season boundaries."""
    if team not in streak or season_seen.get(team) != season:
        return 0
    return streak[team]


def compute_wins_rolling(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute number of wins in the last N games entering each game.

    Complements ELO (which is slow-moving) with a more reactive recent
    form signal.  Season boundaries are respected: wins do not carry over.

    Args:
        games: DataFrame with columns HOME, AWAY, HOME_PTS, AWAY_PTS,
               DATE, SEASON.
        window: Number of most recent games to consider (default 10).

    Returns:
        Input DataFrame sorted by DATE with added columns:
            HOME_WINS_L{window} - wins in last {window} games for home team
            AWAY_WINS_L{window} - wins in last {window} games for away team

        Values are 0 at season start.  When fewer than {window} games have
        been played, wins are counted over however many games exist.
        Safe to use as model features (pre-game only, no leakage).
    """
    _validate_columns(games)

    games = games.sort_values("DATE").reset_index(drop=True)

    # Track per-team result history within the current season
    history: dict[str, list[int]] = {}      # team -> list of recent win (1) or loss (0)
    season_seen: dict[str, int] = {}

    home_wins: list[int] = []
    away_wins: list[int] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        season = row["SEASON"]

        # Reset history at season boundary
        if season_seen.get(home) != season:
            history[home] = []
            season_seen[home] = season
        if season_seen.get(away) != season:
            history[away] = []
            season_seen[away] = season

        home_wins.append(sum(history[home][-window:]))
        away_wins.append(sum(history[away][-window:]))

        home_won = row["HOME_PTS"] > row["AWAY_PTS"]
        history[home].append(1 if home_won else 0)
        history[away].append(0 if home_won else 1)

    col = f"_L{window}"
    result = games.copy()
    result[f"HOME_WINS{col}"] = home_wins
    result[f"AWAY_WINS{col}"] = away_wins
    return result


def _validate_columns(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(
            f"games DataFrame is missing required columns: {sorted(missing)}"
        )

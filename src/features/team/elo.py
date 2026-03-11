"""ELO rating system for NBA team strength estimation.

Refactored from notebooks/feature-engineering.ipynb.

Key design decisions (preserved from original notebook):
- Home court advantage = 69 ELO points (~60% win prob for equal teams at home)
- MOV-adjusted K-factor: blowouts move ratings more than close games
- Season regression: 75% carry-forward + 25% regression to 1505 at each new season
- Starting ELO = 1500 for teams with no prior history
- Season regression target = 1505 (not 1500) to give a slight edge to teams that
  completed the prior season, which skews toward playoff-caliber teams
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_COURT_ADVANTAGE: float = 69.0   # ELO points; implies ~60% home win prob
STARTING_ELO: float = 1500.0
SEASON_REGRESSION_TARGET: float = 1505.0
SEASON_REGRESSION_WEIGHT: float = 0.25  # 25% toward mean, 75% carry-forward
BASE_K: float = 20.0


# ---------------------------------------------------------------------------
# Core ELO functions
# ---------------------------------------------------------------------------

def win_probs(
    home_elo: float,
    away_elo: float,
    home_court_advantage: float = HOME_COURT_ADVANTAGE,
) -> tuple[float, float]:
    """Return (home_win_prob, away_win_prob) using the ELO logistic curve.

    The home court advantage is baked in as extra ELO points for the home
    team.  The two probabilities always sum to exactly 1.0.
    """
    h = 10 ** (home_elo / 400)
    r = 10 ** (away_elo / 400)
    a = 10 ** (home_court_advantage / 400)
    denom = r + a * h
    return (a * h / denom), (r / denom)


def elo_k(mov: float, elo_diff: float) -> float:
    """Return the MOV-adjusted K-factor for an ELO update.

    Args:
        mov: Margin of victory from the home team's perspective
             (positive = home win, negative = away win).
        elo_diff: home_elo - away_elo *before* the game.

    The (|MOV|+3)^0.8 exponent grows with margin but sub-linearly, so a
    40-point blowout does not move ratings 10x more than a 4-point game.
    The elo_diff denominator term dampens movement when the stronger team
    beats the weaker team by a large margin (expected result, less info).
    """
    if mov > 0:
        multiplier = (mov + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)
    else:
        multiplier = (-mov + 3) ** 0.8 / (7.5 + 0.006 * (-elo_diff))
    return BASE_K * multiplier


def update_elo(
    home_score: float,
    away_score: float,
    home_elo: float,
    away_elo: float,
    home_court_advantage: float = HOME_COURT_ADVANTAGE,
) -> tuple[float, float]:
    """Update ELO ratings given a completed game result.

    Args:
        home_score: Points scored by the home team.
        away_score: Points scored by the away team.
        home_elo: Home team ELO *before* the game.
        away_elo: Away team ELO *before* the game.
        home_court_advantage: ELO points added for home team.

    Returns:
        (new_home_elo, new_away_elo)
    """
    home_prob, away_prob = win_probs(home_elo, away_elo, home_court_advantage)
    home_win = 1.0 if home_score > away_score else 0.0
    away_win = 1.0 - home_win
    k = elo_k(home_score - away_score, home_elo - away_elo)
    return (
        home_elo + k * (home_win - home_prob),
        away_elo + k * (away_win - away_prob),
    )


def apply_season_regression(elo: float) -> float:
    """Apply inter-season regression to the mean.

    Called once per team at the start of each new season.  Pulls ratings
    75% toward their prior value and 25% toward 1505, preventing extreme
    ratings from prior seasons from dominating early-season predictions.
    """
    return (
        (1 - SEASON_REGRESSION_WEIGHT) * elo
        + SEASON_REGRESSION_WEIGHT * SEASON_REGRESSION_TARGET
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    games: pd.DataFrame,
    home_court_advantage: float = HOME_COURT_ADVANTAGE,
) -> pd.DataFrame:
    """Compute pre-game and post-game ELO ratings for every game.

    Processes games in chronological order (by DATE), maintaining a running
    ELO state dict.  Season regression is applied automatically when a team
    appears in a new season for the first time.

    Args:
        games: DataFrame with columns HOME, AWAY, HOME_PTS, AWAY_PTS,
               DATE, SEASON.  DATE must be sortable (str YYYY-MM-DD or
               datetime).  SEASON is the integer season year (e.g. 2024).
        home_court_advantage: ELO point boost for the home team.

    Returns:
        The input DataFrame sorted by DATE with four additional columns:
            HOME_ELO   – home team ELO *entering* the game  (no leakage)
            AWAY_ELO   – away team ELO *entering* the game  (no leakage)
            HOME_AFTER – home team ELO *after* the game
            AWAY_AFTER – away team ELO *after* the game

    Data integrity guarantee:
        HOME_ELO and AWAY_ELO reflect only games that have already been
        played before this game's DATE.  Safe to use as model features.
    """
    _validate_columns(games)

    games = games.sort_values("DATE").reset_index(drop=True)

    elo_state: dict[str, float] = {}     # team -> current ELO
    season_seen: dict[str, int] = {}     # team -> last season processed

    home_elo_pre: list[float] = []
    away_elo_pre: list[float] = []
    home_elo_post: list[float] = []
    away_elo_post: list[float] = []

    for _, row in games.iterrows():
        home = row["HOME"]
        away = row["AWAY"]
        season = row["SEASON"]

        home_elo = _get_pre_game_elo(home, season, elo_state, season_seen)
        away_elo = _get_pre_game_elo(away, season, elo_state, season_seen)

        home_elo_pre.append(home_elo)
        away_elo_pre.append(away_elo)

        new_home, new_away = update_elo(
            row["HOME_PTS"], row["AWAY_PTS"],
            home_elo, away_elo,
            home_court_advantage,
        )

        home_elo_post.append(new_home)
        away_elo_post.append(new_away)

        elo_state[home] = new_home
        elo_state[away] = new_away
        season_seen[home] = season
        season_seen[away] = season

    result = games.copy()
    result["HOME_ELO"] = home_elo_pre
    result["AWAY_ELO"] = away_elo_pre
    result["HOME_AFTER"] = home_elo_post
    result["AWAY_AFTER"] = away_elo_post
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(games: pd.DataFrame) -> None:
    required = {"HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(
            f"games DataFrame is missing required columns: {sorted(missing)}"
        )


def _get_pre_game_elo(
    team: str,
    season: int,
    elo_state: dict[str, float],
    season_seen: dict[str, int],
) -> float:
    """Return the appropriate pre-game ELO for a team, applying season
    regression if the team is appearing in a new season for the first time."""
    if team not in elo_state:
        return STARTING_ELO
    if season_seen.get(team) != season:
        return apply_season_regression(elo_state[team])
    return elo_state[team]

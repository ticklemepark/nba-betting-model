"""Edge calculator: compare model output to Underdog Fantasy implied probabilities.

For player props, we convert the model's quantile predictions [low, median, high]
into P(stat > line) using a normal approximation.  For game outcomes, edge is
simply model_prob_home − underdog_implied_prob_home.

Usage:
    from src.betting.edge_calculator import screen_prop_picks, screen_game_picks

    prop_picks = screen_prop_picks(player_df, models, prop_lines)
    game_picks = screen_game_picks(game_df, outcome_model, game_lines)
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# Minimum edge to consider a pick interesting.
# These are conservative defaults; override per call.
_DEFAULT_MIN_EDGE_GAME  = 0.04   # 4 percentage points vs. implied prob
_DEFAULT_MIN_EDGE_PROP  = 0.04   # 4 percentage points vs. implied prob


# ---------------------------------------------------------------------------
# Pick dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PropPick:
    player_name:       str
    team:              str
    opp:               str
    game_id:           str
    stat:              str
    direction:         str    # "over" or "under"
    line:              float
    model_median:      float
    model_low:         float
    model_high:        float
    model_prob_over:   float  # P(stat > line) from quantile interval
    underdog_prob_over: float # Underdog's implied P(over)
    edge:              float  # model_prob_over − underdog_prob_over (positive = OVER edge)
    game_date:         date


@dataclass
class GamePick:
    game_id:              str
    home_team:            str
    away_team:            str
    direction:            str    # "home" or "away"
    model_prob_home:      float
    underdog_prob_home:   float
    edge:                 float  # |model_prob_home − underdog_prob_home|, signed toward direction
    game_date:            date


# ---------------------------------------------------------------------------
# Core probability functions
# ---------------------------------------------------------------------------

def prob_over_from_quantiles(
    median: float,
    low: float,
    high: float,
    line: float,
) -> float:
    """Estimate P(stat > line) using a normal approximation.

    The quantile model gives us:
        low  ≈ 10th percentile  (Φ⁻¹(0.10) ≈ −1.282)
        high ≈ 90th percentile  (Φ⁻¹(0.90) ≈ +1.282)

    We estimate σ from the interquartile-like range (high − low):
        σ = (high − low) / (2 × 1.282)

    Then P(stat > line) = 1 − Φ((line − median) / σ).

    If σ ≤ 0 (degenerate model), falls back to step function at median.
    """
    sigma = (high - low) / 2.564  # 2 × Φ⁻¹(0.90)
    if sigma <= 0:
        return 1.0 if line < median else 0.0
    return float(1.0 - stats.norm.cdf((line - median) / sigma))


def calculate_prop_edge(
    model_median: float,
    model_low: float,
    model_high: float,
    underdog_line: float,
    underdog_over_prob: float,
    min_edge: float = _DEFAULT_MIN_EDGE_PROP,
) -> float | None:
    """Compute edge for a single player prop.

    Returns:
        Signed edge (model_prob_over − underdog_prob_over).
        Positive → OVER edge; negative → UNDER edge.
        Returns None if |edge| < min_edge.
    """
    model_prob = prob_over_from_quantiles(model_median, model_low, model_high, underdog_line)
    edge = model_prob - underdog_over_prob
    return edge if abs(edge) >= min_edge else None


def calculate_game_edge(
    model_prob_home: float,
    underdog_home_prob: float,
    min_edge: float = _DEFAULT_MIN_EDGE_GAME,
) -> float | None:
    """Compute edge for a game outcome (home win probability).

    Returns:
        Signed edge (model_prob_home − underdog_prob_home).
        Positive → HOME edge; negative → AWAY edge.
        Returns None if |edge| < min_edge.
    """
    edge = model_prob_home - underdog_home_prob
    return edge if abs(edge) >= min_edge else None


# ---------------------------------------------------------------------------
# Batch screeners
# ---------------------------------------------------------------------------

def screen_prop_picks(
    player_df: pd.DataFrame,
    models: dict,  # stat (str) → PlayerPropModel
    prop_lines: list,  # list[UnderdogPropLine]
    min_edge: float = _DEFAULT_MIN_EDGE_PROP,
) -> list[PropPick]:
    """Match each Underdog prop line to our model prediction and filter by edge.

    Args:
        player_df:   Today's player feature rows (output of build_player_features).
                     Must contain PLAYER_NAME (or PLAYER_ID) and the relevant
                     feature columns.
        models:      Dict mapping stat → trained PlayerPropModel.
        prop_lines:  List of UnderdogPropLine from fetch_prop_lines().
        min_edge:    Minimum |edge| to include (default 4 pp).

    Returns:
        List of PropPick with |edge| ≥ min_edge, sorted by |edge| descending.
    """
    from src.models.player_props import _get_feature_cols  # avoid circular at module level

    picks: list[PropPick] = []

    if player_df.empty:
        log.warning("screen_prop_picks: player_df is empty.")
        return picks

    # Build a name→row index for quick lookup
    name_col = "PLAYER_NAME" if "PLAYER_NAME" in player_df.columns else None
    if name_col is None:
        log.warning("player_df has no PLAYER_NAME column — cannot match prop lines.")
        return picks

    df_indexed = player_df.set_index(name_col)

    for pl in prop_lines:
        stat = pl.stat
        if stat not in models:
            log.debug("No model for stat %s — skipping %s.", stat, pl.player_name)
            continue

        model = models[stat]
        name  = pl.player_name

        if name not in df_indexed.index:
            log.debug("Player %s not in today's feature df — skipping.", name)
            continue

        row = df_indexed.loc[name]
        # If multiple rows for the same player (shouldn't happen for 1-day feature df),
        # take the last.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        feat_cols = _get_feature_cols(stat, list(row.index))
        if not feat_cols:
            log.debug("No feature cols for %s/%s.", name, stat)
            continue

        X = pd.DataFrame([row[feat_cols].values], columns=feat_cols)
        preds = model.predict(X)
        median = float(preds["median"][0])
        low    = float(preds["low"][0])
        high   = float(preds["high"][0])

        edge = calculate_prop_edge(
            median, low, high,
            pl.line, pl.over_payout,
            min_edge=min_edge,
        )
        if edge is None:
            continue

        model_prob_over = prob_over_from_quantiles(median, low, high, pl.line)
        direction = "over" if edge > 0 else "under"

        picks.append(PropPick(
            player_name        = name,
            team               = pl.team,
            opp                = pl.opp,
            game_id            = pl.game_id,
            stat               = stat,
            direction          = direction,
            line               = pl.line,
            model_median       = median,
            model_low          = low,
            model_high         = high,
            model_prob_over    = model_prob_over,
            underdog_prob_over = pl.over_payout,
            edge               = edge,
            game_date          = pl.game_date,
        ))

    picks.sort(key=lambda p: abs(p.edge), reverse=True)
    log.info("screen_prop_picks: %d picks with |edge| >= %.2f.", len(picks), min_edge)
    return picks


def screen_game_picks(
    game_df: pd.DataFrame,
    model,  # GameOutcomeModel
    game_lines: list,  # list[UnderdogGameLine]
    min_edge: float = _DEFAULT_MIN_EDGE_GAME,
) -> list[GamePick]:
    """Match each Underdog game line to our model prediction and filter by edge.

    Args:
        game_df:     Today's game feature rows (output of build_game_features).
                     Must contain HOME and AWAY team columns.
        model:       Trained GameOutcomeModel (or EnsembleModel).
        game_lines:  List of UnderdogGameLine from fetch_game_lines().
        min_edge:    Minimum |edge| to include (default 4 pp).

    Returns:
        List of GamePick with |edge| ≥ min_edge, sorted by |edge| descending.
    """
    picks: list[GamePick] = []

    if game_df.empty:
        log.warning("screen_game_picks: game_df is empty.")
        return picks

    # Normalise team columns to uppercase for matching
    home_col = next((c for c in game_df.columns if c.upper() == "HOME"), None)
    away_col = next((c for c in game_df.columns if c.upper() == "AWAY"), None)
    if home_col is None or away_col is None:
        log.warning("game_df missing HOME or AWAY column — cannot match game lines.")
        return picks

    probs = model.predict_proba(game_df)
    # predict_proba returns shape (n,) for EnsembleModel or (n,2) for GameOutcomeModel
    if probs.ndim == 2:
        home_probs = probs[:, 1]
    else:
        home_probs = probs

    # Build a (home, away) → prob map
    prob_map: dict[tuple[str, str], float] = {}
    for i, (_, row) in enumerate(game_df.iterrows()):
        key = (str(row[home_col]).upper(), str(row[away_col]).upper())
        prob_map[key] = float(home_probs[i])

    for gl in game_lines:
        key = (gl.home_team.upper(), gl.away_team.upper())
        if key not in prob_map:
            log.debug("Game %s @ %s not in today's feature df — skipping.", gl.away_team, gl.home_team)
            continue

        model_prob_home = prob_map[key]
        edge = calculate_game_edge(model_prob_home, gl.home_payout, min_edge=min_edge)
        if edge is None:
            continue

        direction = "home" if edge > 0 else "away"
        picks.append(GamePick(
            game_id            = gl.game_id,
            home_team          = gl.home_team,
            away_team          = gl.away_team,
            direction          = direction,
            model_prob_home    = model_prob_home,
            underdog_prob_home = gl.home_payout,
            edge               = edge,
            game_date          = gl.game_date,
        ))

    picks.sort(key=lambda p: abs(p.edge), reverse=True)
    log.info("screen_game_picks: %d picks with |edge| >= %.2f.", len(picks), min_edge)
    return picks

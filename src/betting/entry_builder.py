"""Correlation-aware Underdog entry builder.

Picks with positive edge are good individually, but on Underdog you must
combine them into multi-pick entries (2–5 picks).  Because payouts are
exponential, we want to stack *correlated* picks together so that when
one hits, the others tend to hit too.

Correlation heuristics (from CLAUDE.md stacking strategy):
    +0.40  Same game, both OVER counting stats for players on the SAME team
    +0.30  Same game, pace-driven OVER (different teams)
    +0.30  Blowout inverse: star OUT → backup OVER + opponent UNDER
    +0.35  Blowout stack: if game model >70% home win → UNDER on favorite's starters
    −0.20  Same game, opposite direction (one OVER, one UNDER same stat)
     0.00  Different games

These are rule-based scores, not data-derived correlations.  They encode
domain knowledge about how NBA props move together.

Usage:
    from src.betting.entry_builder import build_entries, rank_entries

    entries = build_entries(picks, max_picks=5, max_entries=15)
    ranked  = rank_entries(entries)
    for picks_list, score in ranked[:5]:
        print(score, [p.player_name for p in picks_list])
"""

import itertools
import logging
from typing import Union

from src.betting.edge_calculator import GamePick, PropPick

log = logging.getLogger(__name__)

Pick = Union[PropPick, GamePick]

# Correlation score constants
_SAME_TEAM_OVER_BONUS    = 0.40   # same game, same team, both OVER counting stats
_PACE_OVER_BONUS         = 0.30   # same game, different teams, both OVER
_BLOWOUT_INVERSE_BONUS   = 0.30   # star OUT → backup OVER + opp UNDER
_BLOWOUT_STACK_BONUS     = 0.35   # predicted blowout, UNDER on favourite starters
_SAME_GAME_OPPOSITE_PEN  = -0.20  # same game, head-to-head opposite direction

# Stats where "volume" is driven by game pace (counting stats).
_COUNTING_STATS = {"PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "FG3M", "STL", "BLK", "TOV"}


# ---------------------------------------------------------------------------
# Correlation scoring
# ---------------------------------------------------------------------------

def score_correlation(pick_a: Pick, pick_b: Pick) -> float:
    """Return a correlation score between two picks.

    Score is in [−1, 1].  Higher = more positively correlated
    (i.e. both are more likely to win or lose together).

    Rules applied in priority order:
    1. Game vs. game picks → 0 (different bet types, unrelated)
    2. Different games → 0
    3. Same game, same team, both OVER counting stats → +0.40
    4. Same game, different teams, both OVER counting stats → +0.30
    5. Same game, both picks but one OVER and one UNDER same stat → −0.20
    6. Same game, mixed → 0
    """
    # Blowout-inverse: if one is a GamePick and one is a PropPick in the same game
    if isinstance(pick_a, GamePick) and isinstance(pick_b, PropPick):
        if pick_a.game_id == pick_b.game_id:
            # GamePick predicts a blowout direction; prop is on the underdog's player
            return _blowout_inverse_score(pick_a, pick_b)
        return 0.0

    if isinstance(pick_a, PropPick) and isinstance(pick_b, GamePick):
        if pick_a.game_id == pick_b.game_id:
            return _blowout_inverse_score(pick_b, pick_a)
        return 0.0

    # Both game picks
    if isinstance(pick_a, GamePick) and isinstance(pick_b, GamePick):
        return 0.0  # Independent games

    # Both prop picks
    assert isinstance(pick_a, PropPick) and isinstance(pick_b, PropPick)

    if pick_a.game_id != pick_b.game_id:
        return 0.0  # Different games — no assumed correlation

    # Same game from here on
    same_team = (pick_a.team == pick_b.team)
    both_over = (pick_a.direction == "over" and pick_b.direction == "over")
    both_under = (pick_a.direction == "under" and pick_b.direction == "under")
    opposite  = (pick_a.direction != pick_b.direction)

    a_counting = pick_a.stat in _COUNTING_STATS
    b_counting = pick_b.stat in _COUNTING_STATS

    if same_team and both_over and a_counting and b_counting:
        return _SAME_TEAM_OVER_BONUS

    if same_team and both_under and a_counting and b_counting:
        return _SAME_TEAM_OVER_BONUS  # same logic: both down in a blowout loss

    if not same_team and both_over and a_counting and b_counting:
        return _PACE_OVER_BONUS

    if not same_team and both_under and a_counting and b_counting:
        return _PACE_OVER_BONUS  # both down in a slow game

    if same_team and opposite:
        return _SAME_GAME_OPPOSITE_PEN

    return 0.0


def _blowout_inverse_score(game_pick: GamePick, prop_pick: PropPick) -> float:
    """Score a GamePick + PropPick combination in the same game.

    Positive correlation when:
    - Game model predicts a large win for one side (blowout risk)
    - Prop is UNDER on the favoured team's starters (they'll sit Q4)
    - Prop is OVER on the underdog's key player (desperate volume)
    """
    favourite_team = game_pick.home_team if game_pick.direction == "home" else game_pick.away_team
    underdog_team  = game_pick.away_team if game_pick.direction == "home" else game_pick.home_team

    # Favour's starter getting sat Q4 → UNDER is correlated with blowout
    if prop_pick.team == favourite_team and prop_pick.direction == "under":
        return _BLOWOUT_STACK_BONUS

    # Underdog's player inflating stats in garbage time → OVER is correlated
    if prop_pick.team == underdog_team and prop_pick.direction == "over":
        return _BLOWOUT_INVERSE_BONUS

    return 0.0


def _score_entry(picks: list[Pick]) -> float:
    """Score an entry by mean edge + pairwise correlation bonuses."""
    mean_edge = sum(abs(p.edge) for p in picks) / len(picks)
    corr_bonus = 0.0
    for a, b in itertools.combinations(picks, 2):
        corr_bonus += score_correlation(a, b)
    return mean_edge + 0.5 * corr_bonus


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------

def build_entries(
    picks: list[Pick],
    min_picks: int = 2,
    max_picks: int = 5,
    max_entries: int = 20,
) -> list[list[Pick]]:
    """Generate and rank the top multi-pick entries from a pool of positive-edge picks.

    Args:
        picks:       List of PropPick / GamePick with positive edge.
        min_picks:   Minimum picks per entry (default 2).
        max_picks:   Maximum picks per entry (default 5, Underdog allows up to 6).
        max_entries: Maximum number of entries to return (default 20).

    Returns:
        List of pick-lists, sorted best-first by _score_entry.
    """
    if len(picks) < min_picks:
        log.warning("build_entries: only %d picks available (need %d).", len(picks), min_picks)
        return []

    all_entries: list[tuple[float, list[Pick]]] = []

    # Cap candidate pool to top-30 picks by |edge| to limit combinatorial explosion.
    pool = sorted(picks, key=lambda p: abs(p.edge), reverse=True)[:30]

    for n in range(min_picks, min(max_picks, len(pool)) + 1):
        for combo in itertools.combinations(pool, n):
            score = _score_entry(list(combo))
            all_entries.append((score, list(combo)))

    # Sort descending by score, deduplicate by frozenset of pick ids, take top N.
    all_entries.sort(key=lambda x: x[0], reverse=True)

    seen: set[frozenset] = set()
    top: list[list[Pick]] = []
    for score, entry in all_entries:
        key = frozenset(id(p) for p in entry)
        if key not in seen:
            seen.add(key)
            top.append(entry)
        if len(top) >= max_entries:
            break

    log.info("build_entries: %d entries from %d picks.", len(top), len(picks))
    return top


def rank_entries(
    entries: list[list[Pick]],
) -> list[tuple[list[Pick], float]]:
    """Score and sort entries by _score_entry.

    Returns:
        List of (picks_list, score) tuples, sorted descending by score.
    """
    scored = [(entry, _score_entry(entry)) for entry in entries]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

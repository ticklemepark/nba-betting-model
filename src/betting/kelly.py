"""Kelly criterion bankroll sizing for Underdog Fantasy entries.

Underdog payouts (gross multiplier, i.e. you get this × your stake back):
    2-pick → 3×,  3-pick → 6×,  4-pick → 10×,  5-pick → 20×,  6-pick → 36×

Kelly formula for a bet with net payout b (b = gross_payout − 1) and
win probability p:
    f* = (b·p − (1−p)) / b  =  p − (1−p)/b

We use fractional Kelly (f* × kelly_fraction) for safety.

Usage:
    from src.betting.kelly import fractional_kelly, size_entry, summarise_sizing

    f = fractional_kelly(win_prob=0.55, payout=3.0, kelly_fraction=0.25)
    bet_dollars = size_entry(picks, bankroll=1000)
"""

import logging

import pandas as pd

from src.betting.edge_calculator import GamePick, PropPick

log = logging.getLogger(__name__)

# Underdog's fixed payout table (gross multiplier: you get payout × stake).
UNDERDOG_PAYOUTS: dict[int, float] = {
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 20.0,
    6: 36.0,
}

_DEFAULT_KELLY_FRACTION = 0.25
_MIN_BET_FRACTION       = 0.001   # never bet less than 0.1% of bankroll on one entry
_MAX_BET_FRACTION       = 0.05    # hard cap: never risk more than 5% on one entry


# ---------------------------------------------------------------------------
# Core Kelly function
# ---------------------------------------------------------------------------

def fractional_kelly(
    win_prob: float,
    payout: float,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
) -> float:
    """Compute fractional Kelly bet size as a fraction of bankroll.

    Args:
        win_prob:       P(entry wins) — for multi-pick, product of individual probs.
        payout:         Gross payout multiplier (e.g. 3.0 for a 2-pick entry).
        kelly_fraction: Safety factor; 0.25 = quarter-Kelly (default).

    Returns:
        Fraction of bankroll to bet.  Always in [0, _MAX_BET_FRACTION].
        Returns 0 if the Kelly fraction is negative (edge ≤ 0).
    """
    if win_prob <= 0.0 or payout <= 1.0:
        return 0.0

    net_payout = payout - 1.0           # b in the Kelly formula
    kelly_full = (net_payout * win_prob - (1.0 - win_prob)) / net_payout
    kelly_full = max(0.0, kelly_full)   # clip negative values (no-bet)

    fraction = kelly_full * kelly_fraction
    return min(fraction, _MAX_BET_FRACTION)


# ---------------------------------------------------------------------------
# Entry-level sizing
# ---------------------------------------------------------------------------

def _entry_win_prob(picks: list) -> float:
    """Estimate entry win probability as the product of individual pick probs.

    Each pick's probability of being correct:
    - PropPick: use model_prob_over if direction=="over", else 1-model_prob_over
    - GamePick: use model_prob_home if direction=="home", else 1-model_prob_home

    This assumes independence across picks; correlation adjustment is handled
    in entry_builder before calling this function.
    """
    prob = 1.0
    for pick in picks:
        if isinstance(pick, PropPick):
            p = pick.model_prob_over if pick.direction == "over" else 1.0 - pick.model_prob_over
        elif isinstance(pick, GamePick):
            p = pick.model_prob_home if pick.direction == "home" else 1.0 - pick.model_prob_home
        else:
            p = 0.5
        prob *= max(p, 1e-9)
    return prob


def size_entry(
    picks: list,
    bankroll: float,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
) -> float:
    """Return the dollar amount to bet on a multi-pick entry.

    Args:
        picks:          List of PropPick / GamePick forming the entry.
        bankroll:       Current bankroll in dollars.
        kelly_fraction: Fraction of full-Kelly to use (default 0.25).

    Returns:
        Dollar bet amount.  At least $1 if entry has positive edge,
        capped at _MAX_BET_FRACTION × bankroll.
    """
    n = len(picks)
    if n not in UNDERDOG_PAYOUTS:
        log.warning("size_entry: %d picks — no Underdog payout for this size.", n)
        return 0.0

    payout   = UNDERDOG_PAYOUTS[n]
    win_prob = _entry_win_prob(picks)
    fraction = fractional_kelly(win_prob, payout, kelly_fraction)

    if fraction < _MIN_BET_FRACTION:
        return 0.0

    return round(bankroll * fraction, 2)


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

def summarise_sizing(
    entries: list[list],
    bankroll: float,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
) -> pd.DataFrame:
    """Build a summary DataFrame of bet sizing for a list of entries.

    Args:
        entries:        List of pick lists (each list is one entry).
        bankroll:       Current bankroll in dollars.
        kelly_fraction: Kelly fraction.

    Returns:
        DataFrame with columns:
            entry_size, win_prob, ev, bet_amount, payout_multiplier
        Sorted by ev descending.
    """
    rows = []
    for picks in entries:
        n = len(picks)
        payout   = UNDERDOG_PAYOUTS.get(n, 0.0)
        win_prob = _entry_win_prob(picks)
        ev       = win_prob * payout - 1.0            # expected value per unit staked
        amount   = size_entry(picks, bankroll, kelly_fraction)

        rows.append({
            "entry_size":        n,
            "win_prob":          round(win_prob, 4),
            "ev":                round(ev, 4),
            "payout_multiplier": payout,
            "bet_amount":        amount,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df

"""Combo line divergence detection and exploitation.

Underdog sets combo lines (PRA = PTS+REB+AST, RA = REB+AST) independently
from the component individual lines.  When they disagree, the sum of the
individual lines is usually the more accurate anchor — individual lines
attract more betting action and thus tighter market correction.

Strategy:
  - If sum_individual > combo_line  →  OVER on the combo has edge
  - If combo_line > sum_individual  →  UNDER on the combo has edge

Minimum divergence threshold before treating as exploitable: 1.5 pts.
(1.0 pt divergences are often just rounding to nearest 0.5.)

Supported combos:
  PRA = PTS + REB + AST
  RA  = REB + AST
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
from scipy import stats

from src.data.db import get_cursor

log = logging.getLogger(__name__)

# Combo → component stats
_COMBOS: dict[str, list[str]] = {
    "PRA": ["PTS", "REB", "AST"],
    "RA":  ["REB", "AST"],
}

# Historical typical std-dev for each combo stat (used for P(over) estimate
# when we don't have individual player variance).  Tuned on 2-season actuals.
_COMBO_STD: dict[str, float] = {
    "PRA": 7.0,
    "RA":  3.0,
}

# Minimum divergence (|sum_individual - combo_line|) to flag as exploitable.
MIN_DIVERGENCE = 1.5


@dataclass
class ComboDivergence:
    player_name:    str
    team:           str
    game_id:        str
    game_date:      date
    combo_stat:     str           # 'PRA' or 'RA'
    combo_line:     float         # Underdog's direct combo line
    sum_individual: float         # sum of component lines
    divergence:     float         # sum_individual - combo_line (+ = OVER edge)
    component_lines: dict[str, float]
    direction:      str           # 'over' or 'under'
    edge_pp:        float         # estimated edge in percentage points


def _prob_over(true_center: float, line: float, sigma: float) -> float:
    """P(stat > line) assuming stat ~ N(true_center, sigma)."""
    if sigma <= 0:
        return 1.0 if true_center > line else 0.0
    return float(1.0 - stats.norm.cdf((line - true_center) / sigma))


def detect_divergences(
    prop_lines,           # list[UnderdogPropLine]
    min_divergence: float = MIN_DIVERGENCE,
) -> list[ComboDivergence]:
    """Find combo lines that diverge from their component sum.

    Args:
        prop_lines:      Today's prop lines from fetch_prop_lines().
        min_divergence:  Minimum |divergence| to report (default 1.5).

    Returns:
        List of ComboDivergence objects sorted by |divergence| descending.
    """
    # Index lines by (player_name, stat) → line value
    by_player: dict[str, dict[str, float]] = {}
    game_ids:  dict[str, str] = {}
    teams:     dict[str, str] = {}
    game_dates: dict[str, date] = {}

    for pl in prop_lines:
        name = pl.player_name
        if name not in by_player:
            by_player[name]    = {}
            game_ids[name]     = pl.game_id
            teams[name]        = pl.team
            game_dates[name]   = pl.game_date
        by_player[name][pl.stat] = pl.line

    divergences: list[ComboDivergence] = []

    for name, lines in by_player.items():
        for combo, components in _COMBOS.items():
            if combo not in lines:
                continue
            if not all(c in lines for c in components):
                continue

            combo_line     = lines[combo]
            comp_vals      = {c: lines[c] for c in components}
            sum_individual = sum(comp_vals.values())
            divergence     = sum_individual - combo_line

            if abs(divergence) < min_divergence:
                continue

            direction = "over" if divergence > 0 else "under"

            # Estimate edge: assume true PRA ~ N(sum_individual, sigma)
            sigma      = _COMBO_STD.get(combo, 6.0)
            prob_over  = _prob_over(sum_individual, combo_line, sigma)
            prob_pick  = prob_over if direction == "over" else (1.0 - prob_over)
            edge_pp    = round((prob_pick - 0.5) * 100, 1)

            divergences.append(ComboDivergence(
                player_name     = name,
                team            = teams[name],
                game_id         = game_ids[name],
                game_date       = game_dates[name],
                combo_stat      = combo,
                combo_line      = combo_line,
                sum_individual  = sum_individual,
                divergence      = round(divergence, 1),
                component_lines = comp_vals,
                direction       = direction,
                edge_pp         = edge_pp,
            ))

    divergences.sort(key=lambda d: abs(d.divergence), reverse=True)
    return divergences


def log_divergences_to_db(divergences: list[ComboDivergence]) -> int:
    """Upsert divergence records into line_divergences table.

    Returns number of rows inserted/updated.
    """
    if not divergences:
        return 0

    rows_written = 0
    with get_cursor() as cur:
        for d in divergences:
            cur.execute(
                """
                INSERT INTO line_divergences
                    (game_date, player_name, team, combo_stat,
                     combo_line, sum_individual, divergence, component_lines)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_name, combo_stat, game_date)
                DO UPDATE SET
                    combo_line      = EXCLUDED.combo_line,
                    sum_individual  = EXCLUDED.sum_individual,
                    divergence      = EXCLUDED.divergence,
                    component_lines = EXCLUDED.component_lines,
                    logged_at       = NOW()
                """,
                (
                    d.game_date, d.player_name, d.team, d.combo_stat,
                    d.combo_line, d.sum_individual, d.divergence,
                    json.dumps(d.component_lines),
                ),
            )
            rows_written += 1

    log.info("Logged %d combo line divergences to DB.", rows_written)
    return rows_written


def settle_divergences(game_date: date, actual_stats: dict[str, dict[str, float]]) -> int:
    """Fill in actual_stat and closer_to for a settled game date.

    Args:
        game_date:    The game date to settle.
        actual_stats: {player_name: {stat: actual_value}} from settle_results.

    Returns:
        Number of rows updated.
    """
    updated = 0
    with get_cursor() as cur:
        cur.execute(
            "SELECT id, player_name, combo_stat, combo_line, sum_individual "
            "FROM line_divergences WHERE game_date = %s AND actual_stat IS NULL",
            (game_date,),
        )
        rows = cur.fetchall()

    with get_cursor() as cur:
        for row_id, player_name, combo_stat, combo_line, sum_individual in rows:
            player_stats = actual_stats.get(player_name, {})

            combo_def = _COMBOS.get(combo_stat, [])
            if not all(c in player_stats for c in combo_def):
                continue

            actual = sum(player_stats[c] for c in combo_def)
            err_combo  = abs(actual - float(combo_line))
            err_sum    = abs(actual - float(sum_individual))

            if err_sum < err_combo:
                closer_to = "sum_individual"
            elif err_combo < err_sum:
                closer_to = "combo"
            else:
                closer_to = "tie"

            cur.execute(
                "UPDATE line_divergences SET actual_stat=%s, closer_to=%s "
                "WHERE id=%s",
                (actual, closer_to, row_id),
            )
            updated += 1

    if updated:
        log.info("Settled %d divergence records for %s.", updated, game_date)
    return updated


def get_divergence_summary() -> pd.DataFrame:
    """Return accuracy summary: how often sum_individual beats combo_line."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT combo_stat,
                   COUNT(*)                                             AS n_total,
                   SUM(CASE WHEN actual_stat IS NOT NULL THEN 1 END)   AS n_settled,
                   SUM(CASE WHEN closer_to='sum_individual' THEN 1 END) AS n_sum_wins,
                   SUM(CASE WHEN closer_to='combo'          THEN 1 END) AS n_combo_wins,
                   SUM(CASE WHEN closer_to='tie'            THEN 1 END) AS n_ties,
                   AVG(ABS(divergence))                                 AS avg_divergence
            FROM line_divergences
            GROUP BY combo_stat
            ORDER BY combo_stat
            """
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    return pd.DataFrame(rows, columns=cols)

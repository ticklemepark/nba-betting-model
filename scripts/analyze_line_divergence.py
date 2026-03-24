#!/usr/bin/env python3
"""Analyze combo line divergences (PRA, RA) vs sum of component lines.

Shows:
  1. Today's divergences (live Underdog lines)
  2. Historical divergence log from DB
  3. Accuracy: how often sum_individual was closer to actual vs combo line

Usage:
    python scripts/analyze_line_divergence.py
    python scripts/analyze_line_divergence.py --date 2026-03-16
    python scripts/analyze_line_divergence.py --min-div 1.0   # lower threshold
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.line_divergence import (
    MIN_DIVERGENCE,
    detect_divergences,
    get_divergence_summary,
)
from src.data.db import get_cursor
from src.data.scrapers.underdog import fetch_prop_lines


def _sep(char="-", width=70):
    return char * width


def print_divergences_from_db(game_date=None, min_div=1.0):
    params = [min_div]
    where  = "WHERE ABS(divergence) >= %s"
    if game_date:
        where += " AND game_date = %s"
        params.append(game_date)

    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT game_date, player_name, team, combo_stat,
                   sum_individual, combo_line, divergence,
                   component_lines, actual_stat, closer_to
            FROM line_divergences
            {where}
            ORDER BY game_date DESC, ABS(divergence) DESC
            """,
            params,
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    if not rows:
        print("  No divergences found in DB.")
        return

    print(f"  {'DATE':>12}  {'PLAYER':<28}  {'COMBO':>5}  "
          f"{'SUM':>6}  {'LINE':>6}  {'DIV':>5}  {'DIR':>9}  "
          f"{'ACTUAL':>7}  {'CLOSER'}")
    print(_sep())
    for row in rows:
        d = dict(zip(cols, row))
        div   = float(d["divergence"])
        direction = "OVER" if div > 0 else "UNDER"
        actual_str = f"{d['actual_stat']:.1f}" if d["actual_stat"] is not None else "—"
        closer_str = d["closer_to"] or "—"
        print(f"  {str(d['game_date']):>12}  {d['player_name']:<28}  "
              f"{d['combo_stat']:>5}  "
              f"{float(d['sum_individual']):>6.1f}  {float(d['combo_line']):>6.1f}  "
              f"{div:>+5.1f}  {direction:>9}  "
              f"{actual_str:>7}  {closer_str}")
    print()


def print_accuracy_summary():
    df = get_divergence_summary()
    if df.empty or df["n_settled"].sum() == 0:
        print("  No settled divergences yet — check back after games complete.")
        return

    print(_sep())
    print(f"  ACCURACY: sum_individual vs combo_line (settled games)")
    print(_sep())
    for _, row in df.iterrows():
        n_s = int(row["n_settled"] or 0)
        if n_s == 0:
            continue
        n_sw  = int(row["n_sum_wins"]   or 0)
        n_cw  = int(row["n_combo_wins"] or 0)
        n_tie = int(row["n_ties"]       or 0)
        pct   = n_sw / n_s * 100 if n_s else 0
        print(f"  {row['combo_stat']:>5}  settled={n_s}  "
              f"sum_wins={n_sw} ({pct:.0f}%)  combo_wins={n_cw}  ties={n_tie}  "
              f"avg_div={float(row['avg_divergence']):.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Combo line divergence analysis.")
    parser.add_argument("--date",    type=str,   default=None,
                        help="Game date YYYY-MM-DD to filter (default: all)")
    parser.add_argument("--min-div", type=float, default=1.0,
                        help="Minimum |divergence| to show (default 1.0)")
    parser.add_argument("--live",    action="store_true",
                        help="Fetch live Underdog lines and show today's divergences")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    sep = "=" * 70

    print()
    print(f"  NBA COMBO LINE DIVERGENCE REPORT")
    print()

    if args.live:
        print(sep)
        print("  LIVE LINES — TODAY'S DIVERGENCES")
        print(sep)
        try:
            lines = fetch_prop_lines()
            divs  = detect_divergences(lines, min_divergence=args.min_div)
            if not divs:
                print(f"  No divergences >= {args.min_div} found in live lines.")
            else:
                print(f"  {'PLAYER':<28}  {'COMBO':>5}  {'SUM':>6}  {'LINE':>6}  "
                      f"{'DIV':>5}  {'DIR':>9}  {'EDGE_PP':>8}")
                print(_sep())
                for d in divs:
                    direction = "OVER" if d.divergence > 0 else "UNDER"
                    comp_str  = "+".join(f"{k}={v}" for k, v in d.component_lines.items())
                    print(f"  {d.player_name:<28}  {d.combo_stat:>5}  "
                          f"{d.sum_individual:>6.1f}  {d.combo_line:>6.1f}  "
                          f"{d.divergence:>+5.1f}  {direction:>9}  {d.edge_pp:>+7.1f}pp"
                          f"  [{comp_str}]")
        except Exception as e:
            print(f"  Failed to fetch live lines: {e}")
        print()

    print(sep)
    print("  HISTORICAL DIVERGENCE LOG (DB)")
    print(sep)
    print_divergences_from_db(game_date=target_date, min_div=args.min_div)

    print(sep)
    print_accuracy_summary()
    print(sep)
    print()
    print("  NOTE: After ~2 weeks of data, check if sum_wins > 60% consistently.")
    print("  If so, divergences >= 1.5 pts are exploitable as UNDER/OVER combo picks.")
    print()


if __name__ == "__main__":
    main()

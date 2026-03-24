#!/usr/bin/env python3
"""P&L summary report for Underdog Fantasy paper-trading / live bets.

Pulls settled entries from the database and prints:
  1. Overall headline numbers (wagered / returned / net P&L / ROI / win rate).
  2. Break-down by entry size vs. Underdog break-even and target win rates.
  3. Optional day-by-day table (--by-day).

Usage:
    python scripts/pnl_report.py                        # all history
    python scripts/pnl_report.py --start 2026-01-01    # from date
    python scripts/pnl_report.py --end   2026-03-08    # up to date
    python scripts/pnl_report.py --by-day               # daily breakdown
    python scripts/pnl_report.py --start 2026-01-01 --end 2026-03-08 --by-day
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.tracker import get_pnl_summary
from src.data.db import get_cursor

# ---------------------------------------------------------------------------
# Underdog payout table — used to compute break-even and target win rates.
# ---------------------------------------------------------------------------

_PAYOUTS: dict[int, float] = {2: 3.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 36.0}
_TARGET_WIN_RATES: dict[int, float] = {
    2: 0.38,   # target 38 % vs 33.3 % break-even
    3: 0.20,   # target 20 % vs 16.7 %
    4: 0.13,   # target 13 % vs 10.0 %
    5: 0.07,   # target  7 % vs  5.0 %
    6: 0.05,   # target  5 % vs  2.8 %
}


def _breakeven(payout: float) -> float:
    """Minimum win rate to break even at a given gross payout multiplier."""
    return 1.0 / payout


# ---------------------------------------------------------------------------
# Daily breakdown query (not exposed by tracker.get_pnl_summary)
# ---------------------------------------------------------------------------

def _get_daily_pnl(
    start_date: date | None = None,
    end_date:   date | None = None,
) -> pd.DataFrame:
    """Return settled bets aggregated by game_date."""
    params: list = []
    clauses: list[str] = ["status IN ('won','lost')"]
    if start_date:
        clauses.append("game_date >= %s")
        params.append(start_date)
    if end_date:
        clauses.append("game_date <= %s")
        params.append(end_date)

    where = " AND ".join(clauses)

    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT game_date,
                   COUNT(*)                         AS n_entries,
                   SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) AS n_won,
                   SUM(bet_amount)                  AS wagered,
                   SUM(result_amount)               AS net_pnl
            FROM bet_entries
            WHERE {where}
            GROUP BY game_date
            ORDER BY game_date
            """,
            params,
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    if not rows:
        return pd.DataFrame(columns=["game_date", "n_entries", "n_won",
                                     "wagered", "net_pnl", "win_rate",
                                     "returned", "roi"])

    df = pd.DataFrame(rows, columns=cols)
    df["wagered"]   = df["wagered"].astype(float)
    df["net_pnl"]   = df["net_pnl"].astype(float)
    df["win_rate"]  = (df["n_won"] / df["n_entries"]).round(3)
    df["returned"]  = (df["wagered"] + df["net_pnl"]).round(2)
    df["roi"]       = (df["net_pnl"] / df["wagered"].replace(0, float("nan"))).round(4)
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float | None, places: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{places}f}%"


def _dollar(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _sep(char: str = "-", width: int = 62) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _print_headline(summary: dict) -> None:
    n     = summary["n_entries"]
    won   = summary["n_won"]
    lost  = summary["n_lost"]
    wr    = summary["win_rate"]
    waged = summary["total_wagered"]
    ret   = summary["total_returned"]
    pnl   = summary["net_pnl"]
    roi   = summary["roi"]

    print(_sep("="))
    print("  P&L SUMMARY")
    print(_sep("="))
    print(f"  Entries settled : {n:>6,}  ({won} won / {lost} lost)")
    print(f"  Win rate        : {_pct(wr):>7}")
    print(f"  Total wagered   : ${waged:>10,.2f}")
    print(f"  Total returned  : ${ret:>10,.2f}")
    print(f"  Net P&L         : {_dollar(pnl):>11}")
    print(f"  ROI             : {_pct(roi, 2):>7}")
    print()


def _print_by_size(summary: dict) -> None:
    by_size = summary.get("by_entry_size", {})
    if not by_size:
        return

    print(_sep())
    print(f"  {'SIZE':>4}  {'N':>5}  {'WAGERED':>10}  {'NET P&L':>10}  "
          f"{'WIN RATE':>9}  {'BREAK-EVEN':>10}  {'TARGET':>7}  STATUS")
    print(_sep())

    for size in sorted(by_size):
        g   = by_size[size]
        n   = g["n"]
        wag = g["wagered"]
        pnl = g["net_pnl"]
        payout = _PAYOUTS.get(size, 1.0)
        be  = _breakeven(payout)
        tgt = _TARGET_WIN_RATES.get(size)

        # We don't store per-size win counts in by_entry_size — re-query would
        # be heavy; show P&L sign as proxy.
        wr_str = "—"   # not available at this level
        pnl_str = _dollar(pnl)
        status = "[+] positive" if pnl > 0 else ("[-] negative" if pnl < 0 else "flat")

        print(f"  {size:>4}  {n:>5,}  ${wag:>9,.2f}  {pnl_str:>10}  "
              f"{'—':>9}  {_pct(be):>10}  {_pct(tgt):>7}  {status}")

    print()


def _print_daily(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No daily data.")
        return

    print(_sep())
    print(f"  {'DATE':>12}  {'N':>4}  {'WON':>4}  {'WIN%':>6}  "
          f"{'WAGERED':>10}  {'NET P&L':>10}  {'ROI':>7}")
    print(_sep())

    cumulative_pnl = 0.0
    for _, row in df.iterrows():
        cumulative_pnl += float(row["net_pnl"])
        wr_str  = _pct(row["win_rate"])
        pnl_str = _dollar(float(row["net_pnl"]))
        roi_str = _pct(row["roi"], 2) if row["roi"] == row["roi"] else "—"   # NaN check
        print(f"  {str(row['game_date']):>12}  {int(row['n_entries']):>4}  "
              f"{int(row['n_won']):>4}  {wr_str:>6}  "
              f"${float(row['wagered']):>9,.2f}  {pnl_str:>10}  {roi_str:>7}")

    print(_sep())
    print(f"  {'CUMULATIVE P&L':>30}  {_dollar(cumulative_pnl):>10}")
    print()


def _print_phase3_comparison(summary: dict) -> None:
    """Compare actuals to Phase 3 backtest targets."""
    print(_sep())
    print("  PHASE 3 BACKTEST TARGETS vs ACTUALS")
    print(_sep())

    wr  = summary["win_rate"]
    roi = summary["roi"]

    # Single-pick accuracy proxy: we don't store this directly, so we skip it.
    print(f"  Single-pick accuracy  target: 55 %+     actual: (settle per-pick to compute)")
    print(f"  Overall ROI           target:  7 %+     actual: {_pct(roi, 2)}")
    print(f"  Overall win rate      target: varies     actual: {_pct(wr)}")

    if roi is not None:
        if roi >= 0.07:
            print()
            print("  [+] ROI is at or above the 7 % Phase 3 target.")
        elif roi >= 0:
            print()
            print("  ⚠  ROI is positive but below the 7 % target — continue shadow mode.")
        else:
            print()
            print("  [-] ROI is negative -- do not go live yet.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Underdog P&L report.")
    parser.add_argument("--start",  type=str, default=None,
                        help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end",    type=str, default=None,
                        help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--by-day", action="store_true",
                        help="Include a day-by-day breakdown table")
    args = parser.parse_args()

    start = date.fromisoformat(args.start) if args.start else None
    end   = date.fromisoformat(args.end)   if args.end   else None

    summary = get_pnl_summary(start_date=start, end_date=end)

    date_range = ""
    if start or end:
        lo = str(start) if start else "all time"
        hi = str(end)   if end   else "today"
        date_range = f" ({lo} → {hi})"

    print()
    print(f"  NBA Betting Model — Underdog Fantasy P&L{date_range}")
    print()

    if summary["n_entries"] == 0:
        print("  No settled entries found for the specified date range.")
        print()
        return

    _print_headline(summary)
    _print_by_size(summary)
    _print_phase3_comparison(summary)

    if args.by_day:
        print(_sep())
        print("  DAILY BREAKDOWN")
        print()
        daily_df = _get_daily_pnl(start_date=start, end_date=end)
        _print_daily(daily_df)

    print(_sep("="))
    print()


if __name__ == "__main__":
    main()

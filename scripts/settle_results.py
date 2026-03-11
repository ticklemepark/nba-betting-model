#!/usr/bin/env python3
"""Settle pending Underdog Fantasy entries after games complete.

For each pending entry logged by the daily pipeline, this script:
  1. Pulls actual player stats from NBA API for the game date.
  2. Determines whether each pick hit (OVER: actual > line, UNDER: actual < line).
  3. Marks the entry won (all picks hit) or lost (any pick missed).

Run this the morning after game day (results are typically in NBA API by 2 AM PT).

Usage:
    python scripts/settle_results.py                      # settle yesterday
    python scripts/settle_results.py --date 2026-03-08   # specific date
    python scripts/settle_results.py --dry-run           # preview, no DB writes
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.tracker import get_pending_entries, settle_entry

# ---------------------------------------------------------------------------
# nba_api imports at module level (required for @patch to work in tests)
# ---------------------------------------------------------------------------
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players as nba_players
    _HAS_NBA_API = True
except ImportError:
    _HAS_NBA_API = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SEASON = "2025-26"

# Underdog canonical stat -> NBA API box score column(s) to sum.
_STAT_COLS: dict[str, list[str]] = {
    "PTS":  ["PTS"],
    "REB":  ["REB"],
    "AST":  ["AST"],
    "PRA":  ["PTS", "REB", "AST"],
    "PR":   ["PTS", "REB"],
    "PA":   ["PTS", "AST"],
    "RA":   ["REB", "AST"],
    "FG3M": ["FG3M"],
    "STL":  ["STL"],
    "BLK":  ["BLK"],
    "TOV":  ["TOV"],
    "FAN":  ["PTS", "REB", "AST", "STL", "BLK"],   # approx fantasy score
    "MIN":  ["MIN"],
}


# ---------------------------------------------------------------------------
# Core stat resolution
# ---------------------------------------------------------------------------

def _get_actual_stat(
    player_name: str,
    stat: str,
    game_date: date,
    season: str,
    _cache: dict,
) -> float | None:
    """Return the actual stat value for a player on game_date via nba_api.

    Results are cached by player name in `_cache` (passed by reference) to
    avoid redundant API calls when settling multiple picks for the same player.

    Returns None if:
      - The stat type is unknown.
      - The player is not found in nba_api.
      - The player did not play on game_date.
      - An API error occurs.
    """
    cols = _STAT_COLS.get(stat)
    if not cols:
        log.warning("Unknown stat '%s' — cannot settle.", stat)
        return None

    if not _HAS_NBA_API:
        log.error("nba_api not installed — cannot fetch actuals.")
        return None

    # Fetch and cache game log on first access for this player.
    if player_name not in _cache:
        try:
            matches = nba_players.find_players_by_full_name(player_name)
            if not matches:
                log.warning("Player not found in nba_api: '%s'", player_name)
                _cache[player_name] = None
                return None

            pid = matches[0]["id"]
            time.sleep(0.6)   # rate limit
            gl  = playergamelog.PlayerGameLog(player_id=pid, season=season, timeout=15)
            df  = gl.get_data_frames()[0]

            if df.empty:
                _cache[player_name] = None
            else:
                df["_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
                _cache[player_name] = df

        except Exception as exc:
            log.warning("nba_api error for '%s': %s", player_name, exc)
            _cache[player_name] = None

    df = _cache[player_name]
    if df is None:
        return None

    rows = df[df["_DATE"] == game_date]
    if rows.empty:
        log.debug("%s did not play on %s.", player_name, game_date)
        return None

    row  = rows.iloc[0]
    vals = [float(row.get(c, 0) or 0) for c in cols if c in row.index]
    return sum(vals) if vals else None


# ---------------------------------------------------------------------------
# Entry settlement
# ---------------------------------------------------------------------------

def settle_date(
    game_date: date,
    season: str = _SEASON,
    dry_run: bool = False,
) -> int:
    """Settle all pending entries for game_date.

    Args:
        game_date: The game date to settle.
        season:    NBA season string (e.g. '2025-26').
        dry_run:   If True, prints results but does not write to DB.

    Returns:
        Number of entries resolved (won or lost).
    """
    pending = get_pending_entries(game_date)
    if pending.empty:
        print(f"  No pending entries for {game_date}.")
        return 0

    entry_refs = pending["entry_ref"].unique()
    print(f"  Found {len(entry_refs)} pending entries for {game_date}.")

    log_cache: dict = {}   # player_name -> DataFrame (shared across picks)
    resolved = 0

    for entry_ref in entry_refs:
        picks   = pending[pending["entry_ref"] == entry_ref]
        results = []
        skipped = False

        for _, pick in picks.iterrows():
            stat      = pick["stat"]
            direction = pick["direction"]
            line      = float(pick["line"]) if pick["line"] is not None else None
            p_name    = pick["player_name"]

            # Game-winner (Rival) picks — skip for now.
            if stat == "GAME" or line is None:
                log.info("Skipping game-winner pick in entry %s (not yet supported).", entry_ref[:8])
                skipped = True
                break

            actual = _get_actual_stat(p_name, stat, game_date, season, log_cache)
            if actual is None:
                log.warning(
                    "Cannot resolve %s %s for %s — skipping entry %s.",
                    p_name, stat, game_date, entry_ref[:8],
                )
                skipped = True
                break

            if direction == "over":
                hit = actual > line
            elif direction == "under":
                hit = actual < line
            else:
                log.warning("Unknown direction '%s' in entry %s.", direction, entry_ref[:8])
                skipped = True
                break

            results.append((p_name, stat, direction, line, actual, hit))

        if skipped or not results:
            continue

        entry_won = all(r[5] for r in results)
        status    = "WON  ✓" if entry_won else "LOST ✗"
        amount    = float(picks.iloc[0]["bet_amount"])

        print(f"\n  Entry {entry_ref[:8]}  {status}  (${amount:.2f})")
        for name, stat, direction, line, actual, hit in results:
            icon = "✓" if hit else "✗"
            print(f"    {icon} {name:<28} {stat:<6} {direction.upper():<6} {line:>5g}  "
                  f"actual={actual:.1f}")

        if not dry_run:
            settle_entry(entry_ref, won=entry_won)

        resolved += 1

    return resolved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Settle pending Underdog entries.")
    parser.add_argument("--date",    type=str, default=None,
                        help="Game date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--season",  type=str, default=_SEASON,
                        help=f"NBA season (default: {_SEASON})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview results without writing to DB")
    args = parser.parse_args()

    game_date = (
        date.fromisoformat(args.date) if args.date
        else date.today() - timedelta(days=1)
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SETTLE RESULTS — {game_date}  (dry_run={args.dry_run})")
    print(sep)

    n = settle_date(game_date, season=args.season, dry_run=args.dry_run)

    print(f"\n  Resolved: {n} entries.")
    if args.dry_run:
        print("  (dry-run — no DB changes made)")
    print()


if __name__ == "__main__":
    main()

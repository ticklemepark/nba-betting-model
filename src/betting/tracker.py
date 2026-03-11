"""Bet tracker: log entries, settle results, and compute P&L.

All bets are stored in the PostgreSQL tables created by migration 003:
    bet_entries  — one row per entry (2–6 pick combo)
    entry_picks  — one row per individual pick within an entry

Usage:
    from src.betting.tracker import log_entry, settle_entry, get_pnl_summary

    # Log a bet before games tip off
    entry_ref = log_entry(picks, bet_amount=50.0, entry_size=3, game_date=date.today())

    # Settle after results are in
    settle_entry(entry_ref, won=True)

    # Review P&L
    summary = get_pnl_summary()
    print(summary)
"""

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Union

import pandas as pd

from src.betting.edge_calculator import GamePick, PropPick
from src.betting.kelly import UNDERDOG_PAYOUTS
from src.data.db import get_cursor

log = logging.getLogger(__name__)

Pick = Union[PropPick, GamePick]


# ---------------------------------------------------------------------------
# Logging bets
# ---------------------------------------------------------------------------

def log_entry(
    picks: list[Pick],
    bet_amount: float,
    game_date: date,
    notes: str = "",
) -> str:
    """Record a new bet entry in the database.

    Args:
        picks:       List of PropPick / GamePick forming the entry (2–6 picks).
        bet_amount:  Dollar amount wagered.
        game_date:   Date of the games in this entry.
        notes:       Optional free-text note (e.g. "injury edge on LeBron").

    Returns:
        entry_ref — unique identifier for this entry (UUID string).
    """
    entry_size = len(picks)
    if entry_size not in UNDERDOG_PAYOUTS:
        raise ValueError(
            f"Invalid entry size {entry_size}. Underdog allows 2–6 picks."
        )

    payout_multiplier = UNDERDOG_PAYOUTS[entry_size]
    entry_ref         = str(uuid.uuid4())
    placed_at         = datetime.now(timezone.utc)

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO bet_entries
                (entry_ref, entry_size, payout_multiplier, bet_amount,
                 placed_at, game_date, status, notes)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s)
            """,
            (entry_ref, entry_size, payout_multiplier, bet_amount,
             placed_at, game_date, notes),
        )

        for pick in picks:
            if isinstance(pick, PropPick):
                cur.execute(
                    """
                    INSERT INTO entry_picks
                        (entry_ref, game_id, player_name, team, stat,
                         direction, line, model_prediction, edge, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry_ref, pick.game_id, pick.player_name, pick.team,
                        pick.stat, pick.direction, pick.line,
                        pick.model_median, pick.edge,
                        pick.model_prob_over if pick.direction == "over"
                            else 1.0 - pick.model_prob_over,
                    ),
                )
            elif isinstance(pick, GamePick):
                cur.execute(
                    """
                    INSERT INTO entry_picks
                        (entry_ref, game_id, player_name, team, stat,
                         direction, line, model_prediction, edge, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry_ref, pick.game_id,
                        None,
                        pick.home_team if pick.direction == "home" else pick.away_team,
                        "GAME", pick.direction, None,
                        pick.model_prob_home, pick.edge,
                        pick.model_prob_home if pick.direction == "home"
                            else 1.0 - pick.model_prob_home,
                    ),
                )

    log.info(
        "Logged entry %s: %d picks, $%.2f, payout %.0f×.",
        entry_ref[:8], entry_size, bet_amount, payout_multiplier,
    )
    return entry_ref


# ---------------------------------------------------------------------------
# Settling bets
# ---------------------------------------------------------------------------

def settle_entry(entry_ref: str, won: bool) -> None:
    """Mark a bet entry as won or lost and record the result amount.

    Args:
        entry_ref: UUID string returned by log_entry().
        won:       True if all picks in the entry hit (entry won).
    """
    with get_cursor() as cur:
        cur.execute(
            "SELECT bet_amount, payout_multiplier FROM bet_entries WHERE entry_ref = %s",
            (entry_ref,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Entry {entry_ref} not found in database.")

        bet_amount        = float(row["bet_amount"])
        payout_multiplier = float(row["payout_multiplier"])
        result_amount     = bet_amount * payout_multiplier if won else -bet_amount
        status            = "won" if won else "lost"
        settled_at        = datetime.now(timezone.utc)

        cur.execute(
            """
            UPDATE bet_entries
            SET status        = %s,
                result_amount = %s,
                settled_at    = %s
            WHERE entry_ref = %s
            """,
            (status, result_amount, settled_at, entry_ref),
        )

    log.info(
        "Settled entry %s: %s, result = %+.2f.",
        entry_ref[:8], status, result_amount,
    )


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def get_pending_entries(game_date: date | None = None) -> pd.DataFrame:
    """Return all unsettled entries, optionally filtered by game date.

    Returns:
        DataFrame with columns from bet_entries joined to entry_picks.
    """
    with get_cursor() as cur:
        if game_date is not None:
            cur.execute(
                """
                SELECT e.entry_ref, e.entry_size, e.payout_multiplier,
                       e.bet_amount, e.placed_at, e.game_date, e.notes,
                       p.player_name, p.team, p.stat, p.direction,
                       p.line, p.edge
                FROM bet_entries e
                JOIN entry_picks p ON p.entry_ref = e.entry_ref
                WHERE e.status = 'pending' AND e.game_date = %s
                ORDER BY e.placed_at, e.entry_ref, p.id
                """,
                (game_date,),
            )
        else:
            cur.execute(
                """
                SELECT e.entry_ref, e.entry_size, e.payout_multiplier,
                       e.bet_amount, e.placed_at, e.game_date, e.notes,
                       p.player_name, p.team, p.stat, p.direction,
                       p.line, p.edge
                FROM bet_entries e
                JOIN entry_picks p ON p.entry_ref = e.entry_ref
                WHERE e.status = 'pending'
                ORDER BY e.placed_at, e.entry_ref, p.id
                """
            )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def get_pnl_summary(
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """Compute P&L summary over a date range (settled entries only).

    Args:
        start_date: Inclusive lower bound (game_date). None → all history.
        end_date:   Inclusive upper bound (game_date). None → today.

    Returns:
        Dict with keys:
            total_wagered, total_returned, net_pnl, roi,
            n_entries, n_won, n_lost, win_rate,
            by_entry_size  (nested dict: size → {wagered, returned, n})
    """
    params: list = []
    where_clauses: list[str] = ["status IN ('won','lost')"]

    if start_date:
        where_clauses.append("game_date >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("game_date <= %s")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses)

    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT entry_size, status, bet_amount, result_amount
            FROM bet_entries
            WHERE {where_sql}
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return {
            "total_wagered": 0.0, "total_returned": 0.0, "net_pnl": 0.0,
            "roi": 0.0, "n_entries": 0, "n_won": 0, "n_lost": 0,
            "win_rate": None, "by_entry_size": {},
        }

    df = pd.DataFrame(rows, columns=["entry_size", "status", "bet_amount", "result_amount"])
    df["bet_amount"]    = df["bet_amount"].astype(float)
    df["result_amount"] = df["result_amount"].astype(float)

    total_wagered  = df["bet_amount"].sum()
    n_won          = int((df["status"] == "won").sum())
    n_lost         = int((df["status"] == "lost").sum())
    n_total        = len(df)
    # result_amount is positive on wins, negative on losses
    net_pnl        = df["result_amount"].sum()
    total_returned = total_wagered + net_pnl
    roi            = net_pnl / total_wagered if total_wagered > 0 else 0.0

    # By entry size
    by_size: dict = {}
    for size, grp in df.groupby("entry_size"):
        by_size[int(size)] = {
            "n":         len(grp),
            "wagered":   round(float(grp["bet_amount"].sum()), 2),
            "net_pnl":   round(float(grp["result_amount"].sum()), 2),
        }

    return {
        "total_wagered":  round(float(total_wagered), 2),
        "total_returned": round(float(total_returned), 2),
        "net_pnl":        round(float(net_pnl), 2),
        "roi":            round(float(roi), 4),
        "n_entries":      n_total,
        "n_won":          n_won,
        "n_lost":         n_lost,
        "win_rate":       round(n_won / n_total, 4) if n_total > 0 else None,
        "by_entry_size":  by_size,
    }

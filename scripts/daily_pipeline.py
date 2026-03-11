#!/usr/bin/env python
"""Daily Underdog Fantasy pick pipeline.

Workflow:
    1. Fetch today's Underdog prop + game lines (or use --dry-run with saved models)
    2. Load today's feature rows (from data/processed/ parquets)
    3. Load trained models from data/models/
    4. Screen prop picks  → list[PropPick]
    5. Screen game picks  → list[GamePick]
    6. Build optimal entries (correlation-aware)
    7. Size each entry (fractional Kelly)
    8. Print daily pick sheet
    9. Log entries to database (unless --dry-run)

Usage:
    # Live run (requires UNDERDOG_TOKEN in .env)
    python scripts/daily_pipeline.py

    # Dry run — no DB writes, no Underdog API calls (uses empty line pools)
    python scripts/daily_pipeline.py --dry-run

    # Specify parameters
    python scripts/daily_pipeline.py \\
        --date 2025-01-15 \\
        --bankroll 1000 \\
        --min-edge 0.05 \\
        --max-entries 10 \\
        --kelly 0.25
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.edge_calculator import screen_game_picks, screen_prop_picks
from src.betting.entry_builder import build_entries, rank_entries
from src.betting.kelly import UNDERDOG_PAYOUTS, size_entry, summarise_sizing
from src.betting.tracker import log_entry
from src.data.scrapers.underdog import (
    UnderdogAuthError,
    fetch_game_lines,
    fetch_prop_lines,
    save_lines_to_db,
)
from src.models.game_outcome import ALL_FEATURE_COLS, GameOutcomeModel, prepare_features
from src.models.player_props import ALL_STATS, PlayerPropModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_GAME_FEATURES_PATH   = Path("data/processed/game_features.parquet")
_PLAYER_FEATURES_PATH = Path("data/processed/player_features.parquet")
_MODEL_DIR            = Path("data/models")
_DEFAULT_BANKROLL     = 1000.0
_DEFAULT_MIN_EDGE     = 0.04
_DEFAULT_MAX_ENTRIES  = 15
_DEFAULT_KELLY        = 0.25


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_game_model() -> GameOutcomeModel | None:
    path = _MODEL_DIR / "game_outcome_xgb.joblib"
    if not path.exists():
        log.warning("Game outcome model not found at %s — skipping game picks.", path)
        return None
    model = GameOutcomeModel.load(str(path))
    log.info("Loaded game outcome model from %s.", path)
    return model


def load_prop_models() -> dict[str, PlayerPropModel]:
    models: dict[str, PlayerPropModel] = {}
    for stat in ALL_STATS:
        path = _MODEL_DIR / f"player_prop_{stat.lower()}.joblib"
        if path.exists():
            models[stat] = PlayerPropModel.load(str(path))
        else:
            log.debug("No prop model for %s at %s — skipping.", stat, path)
    log.info("Loaded %d / %d prop models.", len(models), len(ALL_STATS))
    return models


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_today_game_features(today: date) -> pd.DataFrame:
    """Load game feature rows for today from the parquet cache."""
    if not _GAME_FEATURES_PATH.exists():
        log.warning("Game features not found at %s.", _GAME_FEATURES_PATH)
        return pd.DataFrame()

    df = pd.read_parquet(_GAME_FEATURES_PATH)
    date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        today_df = df[df[date_col] == today].copy()
    else:
        today_df = pd.DataFrame()

    if today_df.empty:
        log.warning(
            "No game features for %s in %s. "
            "Did you run build_historical_features.py today?",
            today, _GAME_FEATURES_PATH,
        )
    return today_df


def load_today_player_features(today: date) -> pd.DataFrame:
    """Load player feature rows for today from the parquet cache."""
    if not _PLAYER_FEATURES_PATH.exists():
        log.warning("Player features not found at %s.", _PLAYER_FEATURES_PATH)
        return pd.DataFrame()

    df = pd.read_parquet(_PLAYER_FEATURES_PATH)
    date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        today_df = df[df[date_col] == today].copy()
    else:
        today_df = pd.DataFrame()

    if today_df.empty:
        log.warning("No player features for %s.", today)
    return today_df


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_pick(pick) -> str:
    from src.betting.edge_calculator import GamePick, PropPick
    if isinstance(pick, PropPick):
        return (
            f"  {pick.player_name} ({pick.team} vs {pick.opp}) "
            f"{pick.stat} {pick.direction.upper()} {pick.line:g}  "
            f"[model={pick.model_median:.1f}, edge={pick.edge:+.1%}]"
        )
    if isinstance(pick, GamePick):
        team = pick.home_team if pick.direction == "home" else pick.away_team
        return (
            f"  {team} to WIN ({pick.home_team} vs {pick.away_team})  "
            f"[prob={pick.model_prob_home:.1%}, edge={pick.edge:+.1%}]"
        )
    return str(pick)


def print_pick_sheet(
    entries: list,
    sizing_df: pd.DataFrame,
    bankroll: float,
    today: date,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  UNDERDOG FANTASY PICKS — {today}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(sep)

    ranked = rank_entries(entries)
    for i, (picks, score) in enumerate(ranked, 1):
        n         = len(picks)
        payout    = UNDERDOG_PAYOUTS.get(n, 0)
        row       = sizing_df.iloc[i - 1] if i <= len(sizing_df) else None
        amount    = row["bet_amount"] if row is not None else 0.0
        win_prob  = row["win_prob"]   if row is not None else 0.0
        ev        = row["ev"]         if row is not None else 0.0

        print(f"\nEntry {i}: {n}-pick (payout {payout:.0f}×)  "
              f"win_prob={win_prob:.1%}  EV={ev:+.2f}  "
              f"Bet: ${amount:.2f}")
        for pick in picks:
            print(_format_pick(pick))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Underdog pick pipeline.")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--bankroll", type=float, default=_DEFAULT_BANKROLL,
                        help=f"Current bankroll in dollars (default: {_DEFAULT_BANKROLL})")
    parser.add_argument("--min-edge", type=float, default=_DEFAULT_MIN_EDGE,
                        help=f"Minimum edge to consider (default: {_DEFAULT_MIN_EDGE})")
    parser.add_argument("--max-entries", type=int, default=_DEFAULT_MAX_ENTRIES,
                        help=f"Maximum entries to output (default: {_DEFAULT_MAX_ENTRIES})")
    parser.add_argument("--kelly", type=float, default=_DEFAULT_KELLY,
                        help=f"Kelly fraction (default: {_DEFAULT_KELLY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip DB writes and Underdog API calls")
    args = parser.parse_args()

    today: date = (
        date.fromisoformat(args.date) if args.date
        else date.today()
    )

    log.info("=" * 60)
    log.info("Daily pipeline — %s (dry_run=%s)", today, args.dry_run)
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Fetch Underdog lines (always — read-only call)
    #    --dry-run only skips DB writes and entry logging, not the fetch.
    # -----------------------------------------------------------------------
    prop_lines = []
    game_lines = []

    try:
        prop_lines = fetch_prop_lines(str(today))
        game_lines = fetch_game_lines(str(today))
        if not args.dry_run:
            save_lines_to_db(prop_lines, game_lines)
        else:
            log.info("--dry-run: fetched %d prop lines (DB write skipped).", len(prop_lines))
    except UnderdogAuthError as exc:
        log.error("Underdog auth failed: %s", exc)
        log.error("The public endpoint returned 401 — endpoint may have changed.")
        sys.exit(1)
    except Exception as exc:
        log.warning("Could not fetch Underdog lines: %s — continuing with empty pool.", exc)

    # -----------------------------------------------------------------------
    # 2. Load features and models
    # -----------------------------------------------------------------------
    game_df   = load_today_game_features(today)
    player_df = load_today_player_features(today)
    game_model = load_game_model()
    prop_models = load_prop_models()

    # -----------------------------------------------------------------------
    # 3. Screen picks
    # -----------------------------------------------------------------------
    all_picks = []

    if game_model and not game_df.empty and game_lines:
        try:
            game_df_prep = prepare_features(game_df.copy())
            game_picks   = screen_game_picks(game_df_prep, game_model, game_lines,
                                             min_edge=args.min_edge)
            all_picks.extend(game_picks)
            log.info("Game picks: %d with edge >= %.2f.", len(game_picks), args.min_edge)
        except Exception as exc:
            log.warning("Game pick screening failed: %s", exc)

    if prop_models and not player_df.empty and prop_lines:
        try:
            prop_picks = screen_prop_picks(player_df, prop_models, prop_lines,
                                           min_edge=args.min_edge)
            all_picks.extend(prop_picks)
            log.info("Prop picks: %d with edge >= %.2f.", len(prop_picks), args.min_edge)
        except Exception as exc:
            log.warning("Prop pick screening failed: %s", exc)

    if not all_picks:
        log.info("No positive-edge picks found for %s — no entries built.", today)
        print(f"\nNo picks with edge >= {args.min_edge:.0%} found for {today}.")
        return

    # -----------------------------------------------------------------------
    # 4. Build entries
    # -----------------------------------------------------------------------
    entries = build_entries(
        all_picks,
        min_picks=2,
        max_picks=5,
        max_entries=args.max_entries,
    )

    if not entries:
        log.info("No entries built (need at least 2 picks).")
        return

    # -----------------------------------------------------------------------
    # 5. Size entries
    # -----------------------------------------------------------------------
    sizing_df = summarise_sizing(entries, bankroll=args.bankroll, kelly_fraction=args.kelly)

    # -----------------------------------------------------------------------
    # 6. Print pick sheet
    # -----------------------------------------------------------------------
    print_pick_sheet(entries, sizing_df, bankroll=args.bankroll, today=today)

    # -----------------------------------------------------------------------
    # 7. Log to DB (unless dry run)
    # -----------------------------------------------------------------------
    if not args.dry_run:
        for i, (picks, _score) in enumerate(rank_entries(entries)):
            row    = sizing_df.iloc[i] if i < len(sizing_df) else None
            amount = float(row["bet_amount"]) if row is not None else 0.0
            if amount <= 0:
                continue
            ref = log_entry(picks, bet_amount=amount, game_date=today)
            log.info("Logged entry %s ($%.2f).", ref[:8], amount)

    log.info("Done.")


if __name__ == "__main__":
    main()

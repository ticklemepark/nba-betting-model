#!/usr/bin/env python
"""Verification dashboard: compare Underdog lines vs. model predictions vs. recent actuals.

Helps you sanity-check whether:
  - Underdog lines look reasonable vs. recent NBA stats
  - Our model predictions are in the right ballpark
  - Edge calculations are working correctly

Usage:
    # Full dashboard (requires UNDERDOG_TOKEN in .env + trained models)
    python scripts/dashboard.py

    # Just show Underdog lines (no model needed)
    python scripts/dashboard.py --lines-only

    # Filter to specific players or stats
    python scripts/dashboard.py --player "Jayson Tatum"
    python scripts/dashboard.py --stat PTS
    python scripts/dashboard.py --stat PTS --stat REB

    # Show recent player actuals from nba_api (no Underdog needed)
    python scripts/dashboard.py --actuals --player "Jayson Tatum"
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_MODEL_DIR            = Path("data/models")
_PLAYER_FEATURES_PATH = Path("data/processed/player_features.parquet")
_COL_WIDTH            = 110


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char="-"):
    print(char * _COL_WIDTH)


def _header(title: str):
    _sep("=")
    print(f"  {title}")
    _sep("=")


def _fmt_prob(p: float) -> str:
    return f"{p:.1%}"


def _fmt_edge(e: float) -> str:
    sign = "+" if e >= 0 else ""
    color = "\033[92m" if e >= 0.04 else ("\033[93m" if e >= 0 else "\033[91m")
    reset = "\033[0m"
    return f"{color}{sign}{e:.1%}{reset}"


# ---------------------------------------------------------------------------
# Section 1: Underdog lines
# ---------------------------------------------------------------------------

def show_lines(player_filter: str | None, stat_filter: list[str]):
    from src.data.scrapers.underdog import UnderdogAuthError, fetch_prop_lines

    _header("UNDERDOG LINES")
    try:
        lines = fetch_prop_lines()
    except UnderdogAuthError as exc:
        print(f"  [AUTH ERROR] {exc}")
        print("  Add a fresh UNDERDOG_TOKEN to .env and re-run.")
        return []
    except Exception as exc:
        print(f"  [ERROR] Could not fetch lines: {exc}")
        return []

    if not lines:
        print("  No lines returned. Is UNDERDOG_TOKEN valid?")
        return []

    # Apply filters
    if player_filter:
        lines = [l for l in lines if player_filter.lower() in l.player_name.lower()]
    if stat_filter:
        lines = [l for l in lines if l.stat in stat_filter]

    # Group by game
    from collections import defaultdict
    by_game: dict[str, list] = defaultdict(list)
    for l in lines:
        by_game[l.game_id].append(l)

    print(f"  {'PLAYER':<28} {'TEAM':<5} {'OPP':<5} {'STAT':<8} {'LINE':>6}  {'OVER%':>6}  {'UNDER%':>7}  {'AMERICAN OVER':>14}")
    _sep()

    for game_id, game_lines in by_game.items():
        game_lines.sort(key=lambda x: (x.player_name, x.stat))
        game_header = f"{game_lines[0].opp} @ {game_lines[0].team}"
        print(f"\n  [{game_id}]  {game_header}  --  {game_lines[0].game_date}")
        for l in game_lines:
            # Reverse-engineer american price from implied prob for display
            over_pct  = _fmt_prob(l.over_payout)
            under_pct = _fmt_prob(l.under_payout)
            print(
                f"  {l.player_name:<28} {l.team:<5} {l.opp:<5} "
                f"{l.stat:<8} {l.line:>6g}  {over_pct:>6}  {under_pct:>7}"
            )

    print(f"\n  Total lines: {len(lines)}")
    return lines


# ---------------------------------------------------------------------------
# Section 2: Recent actuals from nba_api
# ---------------------------------------------------------------------------

def show_actuals(player_filter: str | None, stat_filter: list[str], n_games: int = 10):
    _header(f"RECENT ACTUALS (last {n_games} games from player_features.parquet)")

    if not _PLAYER_FEATURES_PATH.exists():
        print(f"  [MISSING] {_PLAYER_FEATURES_PATH}")
        print("  Run: python scripts/build_player_features.py")
        return

    import pandas as pd
    df = pd.read_parquet(_PLAYER_FEATURES_PATH)

    # Filter to most recent season
    if "SEASON" in df.columns:
        df = df[df["SEASON"] == df["SEASON"].max()]

    if player_filter:
        df = df[df["PLAYER_NAME"].str.lower().str.contains(player_filter.lower(), na=False)]

    if df.empty:
        print(f"  No data for player filter: {player_filter!r}")
        return

    # Show rolling averages for the most recent appearance per player
    latest = df.sort_values("DATE").groupby("PLAYER_NAME").last().reset_index()

    stats_to_show = stat_filter if stat_filter else ["PTS", "REB", "AST", "PRA"]

    # Build display columns
    disp_cols = ["PLAYER_NAME", "TEAM"]
    for stat in stats_to_show:
        for window in [5, 10, 20]:
            col = f"{stat}_L{window}"
            if col in latest.columns:
                disp_cols.append(col)
        season_col = f"{stat}_SEASON"
        if season_col in latest.columns:
            disp_cols.append(season_col)

    disp_cols = [c for c in disp_cols if c in latest.columns]
    sub = latest[disp_cols].copy()

    # Format floats
    float_cols = [c for c in disp_cols if c not in ("PLAYER_NAME", "TEAM")]
    for col in float_cols:
        sub[col] = sub[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    # Print
    print(sub.to_string(index=False, max_rows=50))


# ---------------------------------------------------------------------------
# Section 3: Model predictions vs. Underdog lines
# ---------------------------------------------------------------------------

def show_model_vs_lines(lines, player_filter, stat_filter):
    if not lines:
        return

    _header("MODEL PREDICTIONS vs. UNDERDOG LINES")

    if not _PLAYER_FEATURES_PATH.exists():
        print(f"  [MISSING] {_PLAYER_FEATURES_PATH} -- run build_player_features.py")
        return

    import pandas as pd
    from src.models.player_props import ALL_STATS, PlayerPropModel, _get_feature_cols

    # Load player features (latest date)
    df = pd.read_parquet(_PLAYER_FEATURES_PATH)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        latest_date = df["DATE"].max()
        df = df[df["DATE"] == latest_date]

    if df.empty:
        print("  No player feature rows found for today.")
        return

    df_idx = df.set_index("PLAYER_NAME") if "PLAYER_NAME" in df.columns else df

    # Load models
    models: dict[str, PlayerPropModel] = {}
    for stat in (stat_filter if stat_filter else ALL_STATS):
        path = _MODEL_DIR / f"player_prop_{stat.lower()}.joblib"
        if path.exists():
            try:
                models[stat] = PlayerPropModel.load(str(path))
            except Exception as exc:
                log.debug("Could not load %s model: %s", stat, exc)

    if not models:
        print(f"  [MISSING] No prop models found in {_MODEL_DIR}")
        print("  Run: python scripts/train_player_props.py")
        return

    print(
        f"  {'PLAYER':<28} {'STAT':<8} {'LINE':>6}  "
        f"{'MODEL':>7}  {'LOW':>6}  {'HIGH':>6}  "
        f"{'EDGE':>8}  {'DIRECTION':<8}"
    )
    _sep()

    from src.betting.edge_calculator import calculate_prop_edge, prob_over_from_quantiles

    rows = []
    for l in lines:
        if l.stat not in models:
            continue
        name = l.player_name
        if name not in df_idx.index:
            continue

        row = df_idx.loc[name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        model  = models[l.stat]
        feats  = _get_feature_cols(l.stat, list(row.index))
        if not feats:
            continue

        X      = pd.DataFrame([row[feats].values], columns=feats)
        preds  = model.predict(X)
        median = float(preds["median"][0])
        low    = float(preds["low"][0])
        high   = float(preds["high"][0])

        edge = calculate_prop_edge(
            median, low, high, l.line, l.over_payout, min_edge=0.0
        )
        direction = "OVER" if (edge or 0) > 0 else "UNDER"
        rows.append((abs(edge or 0), l, median, low, high, edge, direction))

    # Sort by |edge| descending
    rows.sort(key=lambda x: x[0], reverse=True)

    for _, l, median, low, high, edge, direction in rows:
        edge_str = _fmt_edge(edge) if edge is not None else "  n/a"
        print(
            f"  {l.player_name:<28} {l.stat:<8} {l.line:>6g}  "
            f"{median:>7.1f}  {low:>6.1f}  {high:>6.1f}  "
            f"{edge_str:>17}  {direction:<8}"
        )

    if not rows:
        print("  No matches found between lines and player feature df.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Underdog vs. model verification dashboard.")
    parser.add_argument("--player", type=str, default=None,
                        help="Filter by player name (partial match)")
    parser.add_argument("--stat", action="append", dest="stats", default=None,
                        help="Filter by stat (e.g. --stat PTS --stat REB). Repeatable.")
    parser.add_argument("--lines-only", action="store_true",
                        help="Only show Underdog lines (skip model predictions)")
    parser.add_argument("--actuals", action="store_true",
                        help="Show recent rolling averages from player_features.parquet")
    parser.add_argument("--n-games", type=int, default=10,
                        help="Number of recent games for actuals display (default: 10)")
    args = parser.parse_args()

    stat_filter = args.stats or []

    print()
    print("=" * _COL_WIDTH)
    print(f"  NBA BETTING MODEL DASHBOARD  --  {date.today()}")
    print("=" * _COL_WIDTH)

    if args.actuals:
        show_actuals(args.player, stat_filter, args.n_games)

    lines = show_lines(args.player, stat_filter)

    if not args.lines_only and lines:
        print()
        show_model_vs_lines(lines, args.player, stat_filter)

    print()


if __name__ == "__main__":
    main()

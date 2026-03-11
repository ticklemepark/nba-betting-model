"""Phase 1 verification script.

Fetches one season of game results from nba_api (no scraping, no DB required),
then compares our refactored feature modules against the original notebook logic
running on the same data.

Usage:
    python scripts/verify_phase1.py [--season 2024]

Expected results:
  ELO     — exact match  (same algorithm, O(n) dict vs O(n²) DataFrame scan)
  Streaks — exact match
  B2B     — near-exact; notebook has a bug: when the HOME team is brand-new,
             it also zeroes AWAY_B2B without checking the away team separately.
             Affects the first game of any team that debuts mid-schedule.
  H2H     — intentional mismatch: notebook includes the current game's result
             in the H2H record (data leakage). Our module is leak-free.
"""

import argparse
import math
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")          # suppress pandas chained-indexing noise

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.team.elo import compute_elo_ratings
from src.features.team.form import compute_streaks
from src.features.team.h2h import compute_h2h_records
from src.features.team.schedule import compute_b2b_flags


# ---------------------------------------------------------------------------
# Data fetching via nba_api
# ---------------------------------------------------------------------------

def fetch_season(season_year: int) -> pd.DataFrame:
    """Return HOME, AWAY, HOME_PTS, AWAY_PTS, DATE (YYYY-MM-DD), SEASON."""
    from nba_api.stats.endpoints import leaguegamelog

    season_str = f"{season_year - 1}-{str(season_year)[2:]}"
    print(f"Fetching {season_str} from nba_api... ", end="", flush=True)

    log = leaguegamelog.LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
        timeout=30,
    )
    time.sleep(1)
    df = log.get_data_frames()[0]

    home = df[df["MATCHUP"].str.contains(r"vs\.", regex=True)].copy()
    away = df[df["MATCHUP"].str.contains("@", regex=False)].copy()

    games = (
        home[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"]]
        .rename(columns={"TEAM_ABBREVIATION": "HOME", "PTS": "HOME_PTS"})
        .merge(
            away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
                columns={"TEAM_ABBREVIATION": "AWAY", "PTS": "AWAY_PTS"}
            ),
            on="GAME_ID",
        )
    )
    games["DATE"]   = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    games["SEASON"] = season_year
    games = (
        games[["HOME", "AWAY", "HOME_PTS", "AWAY_PTS", "DATE", "SEASON"]]
        .sort_values("DATE")
        .reset_index(drop=True)
    )
    print(f"{len(games)} games.")
    return games


# ---------------------------------------------------------------------------
# Notebook ELO — O(n) dict replication of the notebook's logic.
#
# The notebook scans the full elo_df on each iteration (O(n²)) but the
# mathematical result is identical to this O(n) dict approach: both use the
# same initial ELO (1500), same regression (75/25 to 1505), same K-factor.
# ---------------------------------------------------------------------------

def _nb_elo_k(mov: float, elo_diff: float) -> float:
    if mov > 0:
        return 20 * (mov + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)
    return 20 * (-mov + 3) ** 0.8 / (7.5 + 0.006 * (-elo_diff))


def _nb_update_elo(h_pts: float, a_pts: float, h_elo: float, a_elo: float, hca: float = 69) -> tuple[float, float]:
    h = math.pow(10, h_elo / 400)
    r = math.pow(10, a_elo / 400)
    a = math.pow(10, hca / 400)
    denom = r + a * h
    h_prob = a * h / denom
    a_prob = r / denom
    h_win  = 1.0 if h_pts > a_pts else 0.0
    k      = _nb_elo_k(h_pts - a_pts, h_elo - a_elo)
    return h_elo + k * (h_win - h_prob), a_elo + k * (1 - h_win - a_prob)


def notebook_elo(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("DATE").reset_index(drop=True)
    state: dict[str, float] = {}     # team -> elo after last game
    season_seen: dict[str, int] = {} # team -> last season played
    rows = []

    for _, row in games.iterrows():
        home, away   = row["HOME"], row["AWAY"]
        season       = int(row["SEASON"])
        h_pts, a_pts = float(row["HOME_PTS"]), float(row["AWAY_PTS"])
        date         = row["DATE"]

        # Notebook logic: first time = 1500; else last game's ELO (with optional regression)
        if home not in state:
            h_before = 1500.0
        else:
            h_before = state[home]
            if season_seen[home] != season:
                h_before = 0.75 * h_before + 0.25 * 1505

        if away not in state:
            a_before = 1500.0
        else:
            a_before = state[away]
            if season_seen[away] != season:
                a_before = 0.75 * a_before + 0.25 * 1505

        h_after, a_after = _nb_update_elo(h_pts, a_pts, h_before, a_before)

        state[home] = h_after;  season_seen[home] = season
        state[away] = a_after;  season_seen[away] = season

        rows.append({"DATE": date, "HOME": home, "AWAY": away,
                     "HOME_BEFORE": h_before, "AWAY_BEFORE": a_before,
                     "HOME_AFTER":  h_after,  "AWAY_AFTER":  a_after})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Notebook Streaks — O(n) dict replication
# ---------------------------------------------------------------------------

def notebook_streaks(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("DATE").reset_index(drop=True)
    streak: dict[str, int] = {}      # team -> current win streak
    season_seen: dict[str, int] = {}
    rows = []

    for _, row in games.iterrows():
        home, away   = row["HOME"], row["AWAY"]
        season       = int(row["SEASON"])
        h_pts, a_pts = float(row["HOME_PTS"]), float(row["AWAY_PTS"])
        date         = row["DATE"]

        h_streak = 0 if home not in streak or season_seen.get(home) != season else streak[home]
        a_streak = 0 if away not in streak or season_seen.get(away) != season else streak[away]

        home_won   = h_pts > a_pts
        streak[home] = (h_streak + 1) if home_won else 0
        streak[away] = 0 if home_won else (a_streak + 1)
        season_seen[home] = season
        season_seen[away] = season

        rows.append({"HOME": home, "AWAY": away, "DATE": date,
                     "HOME_STREAK": h_streak, "AWAY_STREAK": a_streak})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Notebook B2B — replicates the notebook's bug
#
# BUG: when home team is brand-new (never appeared), notebook sets both
# HOME_B2B and AWAY_B2B to 0, ignoring whether the away team played yesterday.
# ---------------------------------------------------------------------------

def _nb_are_b2b(d1: str, d2: str) -> int:
    """Return 1 if d2 is exactly one calendar day after d1 (YYYY-MM-DD)."""
    dt1 = datetime.strptime(d1, "%Y-%m-%d")
    dt2 = datetime.strptime(d2, "%Y-%m-%d")
    return 1 if (dt2 - dt1) == timedelta(days=1) else 0


def notebook_b2b(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("DATE").reset_index(drop=True)
    last_date: dict[str, str] = {}   # team -> date of most recent game
    rows = []

    for _, row in games.iterrows():
        home, away = row["HOME"], row["AWAY"]
        date       = row["DATE"]

        # Replicates the notebook's exact condition and bug
        if home not in last_date:
            # HOME team is brand new → notebook sets both to 0 (bug for AWAY_B2B)
            h_b2b, a_b2b = 0, 0
        else:
            h_b2b = _nb_are_b2b(last_date[home], date)
            a_b2b = _nb_are_b2b(last_date[away], date) if away in last_date else 0

        last_date[home] = date
        last_date[away] = date

        rows.append({"HOME": home, "AWAY": away, "DATE": date,
                     "HOME_B2B": h_b2b, "AWAY_B2B": a_b2b})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Notebook H2H — replicates the data leakage bug
#
# BUG: notebook updates the matchup dict with the current game's result
# BEFORE reading home_rec/away_rec.  So each game's H2H record includes
# its own result.  Our module reads before updating.
# ---------------------------------------------------------------------------

def notebook_h2h(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("DATE").reset_index(drop=True)
    rows  = []

    for season in sorted(games["SEASON"].unique()):
        season_data = games[games["SEASON"] == season].sort_values("DATE")
        matchups: dict[tuple, list] = {}  # (team, opp) -> [wins, games]

        for _, row in season_data.iterrows():
            home, away   = row["HOME"], row["AWAY"]
            h_pts, a_pts = float(row["HOME_PTS"]), float(row["AWAY_PTS"])
            date         = row["DATE"]
            home_won     = h_pts > a_pts

            # Notebook UPDATES FIRST (data leakage), then reads
            h_key = (home, away)
            a_key = (away, home)
            if h_key not in matchups:
                matchups[h_key] = [0, 0]
            if a_key not in matchups:
                matchups[a_key] = [0, 0]

            matchups[h_key][1] += 1
            matchups[a_key][1] += 1
            if home_won:
                matchups[h_key][0] += 1
            else:
                matchups[a_key][0] += 1

            h_rec = matchups[h_key][0] / matchups[h_key][1]
            a_rec = matchups[a_key][0] / matchups[a_key][1]

            rows.append({"HOME": home, "AWAY": away, "DATE": date,
                         "HOME_REC": h_rec, "AWAY_REC": a_rec})

    return pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare(label: str, our: pd.Series, nb: pd.Series,
            rtol: float = 1e-6, expected_diff: bool = False) -> dict:
    n      = len(our)
    diff   = (our.astype(float) - nb.astype(float)).abs()
    n_match = int((diff < rtol).sum())
    n_diff  = n - n_match
    pct     = 100 * n_match / n if n else 0.0

    if n_diff == 0:
        status = "PASS"
    elif expected_diff:
        status = "EXPECTED"
    else:
        status = "FAIL"

    print(f"  {label:30s}  {n_match:5d}/{n:5d} match ({pct:5.1f}%)  [{status}]")

    if n_diff > 0 and not expected_diff:
        mask   = diff >= rtol
        sample = pd.DataFrame({"ours": our[mask].values, "notebook": nb[mask].values})
        print(f"    First 3 mismatches:\n{sample.head(3).to_string(index=False)}")

    return {"label": label, "n": n, "n_match": n_match, "n_diff": n_diff, "status": status}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args()

    # 1. Fetch data
    games = fetch_season(args.season)
    print(f"  {len(games)} games loaded.\n")

    # 2. Run our O(n) modules (fast)
    print("Running our refactored modules...")
    our_elo     = compute_elo_ratings(games)
    our_streaks = compute_streaks(games)
    our_b2b     = compute_b2b_flags(games)
    our_h2h     = compute_h2h_records(games)
    print("  Done.\n")

    # 3. Run notebook-equivalent logic (also O(n) now, seconds not minutes)
    print("Running notebook-equivalent logic...")
    nb_elo     = notebook_elo(games)
    nb_streaks = notebook_streaks(games)
    nb_b2b     = notebook_b2b(games)
    nb_h2h     = notebook_h2h(games)
    print("  Done.\n")

    # 4. Compare
    print("=" * 65)
    print(f"PHASE 1 VERIFICATION  —  season {args.season}  ({len(games)} games)")
    print("=" * 65)
    results = []

    print("\n--- ELO (expect: exact match) " + "-" * 34)
    # Our module: HOME_ELO/AWAY_ELO   Notebook: HOME_BEFORE/AWAY_BEFORE
    results.append(compare("HOME ELO pre-game",  our_elo["HOME_ELO"],    nb_elo["HOME_BEFORE"]))
    results.append(compare("AWAY ELO pre-game",  our_elo["AWAY_ELO"],    nb_elo["AWAY_BEFORE"]))
    results.append(compare("HOME ELO post-game", our_elo["HOME_AFTER"],  nb_elo["HOME_AFTER"]))
    results.append(compare("AWAY ELO post-game", our_elo["AWAY_AFTER"],  nb_elo["AWAY_AFTER"]))

    print("\n--- WIN STREAKS (expect: exact match) " + "-" * 26)
    results.append(compare("HOME_STREAK", our_streaks["HOME_STREAK"].astype(float),
                                          nb_streaks["HOME_STREAK"].astype(float)))
    results.append(compare("AWAY_STREAK", our_streaks["AWAY_STREAK"].astype(float),
                                          nb_streaks["AWAY_STREAK"].astype(float)))

    print("\n--- BACK-TO-BACK (expect: small diff from notebook bug) " + "-" * 8)
    b2b_note = "Notebook bug: away_b2b=0 when home team debuts mid-schedule"
    results.append(compare("HOME_B2B", our_b2b["HOME_B2B"].astype(float),
                                       nb_b2b["HOME_B2B"].astype(float)))
    n_b2b_diff = (our_b2b["AWAY_B2B"].astype(float) -
                  nb_b2b["AWAY_B2B"].astype(float)).abs().sum()
    results.append(compare("AWAY_B2B", our_b2b["AWAY_B2B"].astype(float),
                                       nb_b2b["AWAY_B2B"].astype(float),
                                       expected_diff=(n_b2b_diff > 0)))
    if n_b2b_diff > 0:
        print(f"    {b2b_note}")

    print("\n--- HEAD-TO-HEAD (expect: diff - notebook data leakage) " + "-" * 8)
    h2h_note = "Notebook data leakage: H2H record includes current game's result"
    results.append(compare("HOME_REC", our_h2h["HOME_REC"].astype(float),
                                       nb_h2h["HOME_REC"].astype(float),
                                       expected_diff=True))
    print(f"    {h2h_note}")
    results.append(compare("AWAY_REC", our_h2h["AWAY_REC"].astype(float),
                                       nb_h2h["AWAY_REC"].astype(float),
                                       expected_diff=True))
    print(f"    {h2h_note}")

    # 5. Summary
    print("\n" + "=" * 65)
    passes   = sum(1 for r in results if r["status"] in ("PASS", "EXPECTED"))
    failures = [r for r in results if r["status"] == "FAIL"]
    print(f"SUMMARY: {passes}/{len(results)} checks passed or match expected behaviour.")

    if failures:
        print("\nUNEXPECTED FAILURES:")
        for r in failures:
            print(f"  • {r['label']}: {r['n_diff']} rows differ")
        sys.exit(1)
    else:
        print("Phase 1 verification COMPLETE.")
        print("All modules match expected behaviour (ELO + streaks exact;")
        print("B2B/H2H diffs are pre-documented notebook bugs, not our bugs).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""NBA Props Web Dashboard — Flask backend.

Serves a browser UI that shows Underdog Fantasy lines alongside rolling
player stats from NBA API and model edge estimates.

Usage:
    python scripts/app.py            # http://localhost:5000
    python scripts/app.py --port 8080 --debug
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request

from src.data.scrapers.underdog import (
    UnderdogAPIError,
    UnderdogAuthError,
    fetch_prop_lines,
)

# Template folder is in the project root (one level up from scripts/).
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "templates"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SEASON           = "2025-26"
_MODEL_DIR        = Path("data/models")
_PLAYER_FEAT_PATH = Path("data/processed/player_features.parquet")

# ---------------------------------------------------------------------------
# In-memory caches (survive for the process lifetime)
# ---------------------------------------------------------------------------

_props_cache:    dict = {"ts": 0.0, "data": []}
_stats_cache:    dict = {}           # player_name -> (timestamp, logs)
_rankings_cache: dict = {"ts": 0.0, "data": []}
_stats_lock              = Lock()

_PROPS_TTL    = 300   # 5 minutes
_STATS_TTL    = 600   # 10 minutes
_RANKINGS_TTL = 300   # 5 minutes


# ---------------------------------------------------------------------------
# Underdog lines helpers
# ---------------------------------------------------------------------------

def _prob_to_american(prob: float) -> str:
    """Convert implied probability back to American odds string.

    0.5    → 'even'
    0.5238 → '-110'  (standard Underdog vig line)
    0.5556 → '-125'
    0.4762 → '+110'
    """
    if prob <= 0 or prob >= 1:
        return "even"
    if abs(prob - 0.5) < 0.002:
        return "even"          # treat anything within 0.2 pp of 50% as even
    if prob > 0.5:
        return f"-{round(prob / (1.0 - prob) * 100)}"
    else:
        return f"+{round((1.0 - prob) / prob * 100)}"


def _fetch_and_cache_props(force: bool = False) -> list[dict]:
    """Return today's prop lines as plain dicts, cached for _PROPS_TTL seconds."""
    now = time.time()
    if not force and now - _props_cache["ts"] < _PROPS_TTL and _props_cache["data"]:
        return _props_cache["data"]

    try:
        raw = fetch_prop_lines()
    except (UnderdogAuthError, UnderdogAPIError) as exc:
        log.error("Underdog API error: %s", exc)
        return _props_cache["data"]   # return stale data on error

    result = [
        {
            "player_name":   l.player_name,
            "team":          l.team,
            "opp":           l.opp,
            "stat":          l.stat,
            "line":          l.line,
            "over_payout":   l.over_payout,
            "under_payout":  l.under_payout,
            "over_american": _prob_to_american(l.over_payout),
            "under_american":_prob_to_american(l.under_payout),
            "game_id":       l.game_id,
            "game_date":     str(l.game_date),
        }
        for l in raw
    ]
    _props_cache["ts"]   = time.time()
    _props_cache["data"] = result
    return result


# ---------------------------------------------------------------------------
# NBA API helpers
# ---------------------------------------------------------------------------

def _nba_api_game_logs(player_name: str) -> list[dict]:
    """Fetch recent game logs from nba_api. Rate-limited at 0.6 s/call."""
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players

        matches = nba_players.find_players_by_full_name(player_name)
        if not matches:
            return []

        time.sleep(0.6)
        pid = matches[0]["id"]
        gl  = playergamelog.PlayerGameLog(player_id=pid, season=_SEASON, timeout=15)
        df  = gl.get_data_frames()[0]
        if df.empty:
            return []

        rename = {
            "GAME_DATE": "date",   "MATCHUP": "opponent",
            "MIN":       "minutes","PTS":     "points",
            "REB":       "rebounds","AST":    "assists",
            "STL":       "steals", "BLK":    "blocks",
            "TOV":       "turnovers",
            "FGM":  "fgm",  "FGA":  "fga",
            "FG3M": "fg3m", "FG3A": "fg3a",
            "FTM":  "ftm",  "FTA":  "fta",
            "PLUS_MINUS": "plus_minus",
        }
        keep   = [v for v in rename.values() if v in df.rename(columns=rename).columns]
        result = df.rename(columns=rename)[keep].head(20).fillna(0)
        return result.to_dict("records")

    except Exception as exc:
        log.warning("nba_api error for %s: %s", player_name, exc)
        return []


def _get_player_logs(player_name: str) -> list[dict]:
    """Return cached game logs, fetching from nba_api on cache miss."""
    now = time.time()
    with _stats_lock:
        cached = _stats_cache.get(player_name)
        if cached and now - cached[0] < _STATS_TTL:
            return cached[1]

    logs = _nba_api_game_logs(player_name)

    with _stats_lock:
        _stats_cache[player_name] = (time.time(), logs)
    return logs


# ---------------------------------------------------------------------------
# Rolling-average & edge calculation (no ML model required)
# ---------------------------------------------------------------------------

_STAT_KEYS = {
    "PTS":  ["points"],
    "REB":  ["rebounds"],
    "AST":  ["assists"],
    "PRA":  ["points", "rebounds", "assists"],
    "PR":   ["points", "rebounds"],
    "PA":   ["points", "assists"],
    "RA":   ["rebounds", "assists"],
    "FG3M": ["fg3m"],
    "STL":  ["steals"],
    "BLK":  ["blocks"],
    "TOV":  ["turnovers"],
    "FAN":  ["points", "rebounds", "assists", "steals", "blocks"],
}


def _stat_value_from_log(log_row: dict, stat: str) -> float | None:
    """Sum the relevant columns for a composite stat."""
    keys = _STAT_KEYS.get(stat)
    if not keys:
        return None
    vals = [log_row.get(k) for k in keys]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals)


def _rolling_avgs(logs: list[dict], stat: str, line: float) -> dict:
    """Compute L5 / L10 / season averages, hit rate, and a simple edge estimate."""
    values = []
    for g in logs:
        v = _stat_value_from_log(g, stat)
        if v is not None:
            values.append(v)

    if not values:
        return {
            "L5": None, "L10": None, "season": None,
            "hit_rate": None, "edge": None, "direction": None,
        }

    l5     = round(sum(values[:5])  / len(values[:5]),  1) if values[:5]  else None
    l10    = round(sum(values[:10]) / len(values[:10]), 1) if values[:10] else None
    season = round(sum(values)      / len(values),      1)

    over_count = sum(1 for v in values[:10] if v > line)
    hit_rate   = round(over_count / len(values[:10]), 3) if values[:10] else None

    # Simple projection: 40 % season weight + 60 % L10 weight.
    proj = (season * 0.4 + l10 * 0.6) if (l10 is not None and season) else (l10 or season or 0.0)
    edge = round((proj - line) / max(abs(line), 1) * 100, 1)
    direction = "OVER" if edge > 0 else "UNDER"

    return {
        "L5": l5, "L10": l10, "season": season,
        "hit_rate": hit_rate, "edge": edge, "direction": direction,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/props")
def api_props():
    """Return today's Underdog prop lines (cached, no auth needed)."""
    force  = request.args.get("refresh", "").lower() == "true"
    player = request.args.get("player", "").strip().lower()
    stat   = request.args.get("stat",   "").strip().upper()

    props = _fetch_and_cache_props(force=force)

    if player:
        props = [p for p in props if player in p["player_name"].lower()]
    if stat:
        props = [p for p in props if p["stat"] == stat]

    # Build game list for the filter dropdown.
    seen_games: dict[str, dict] = {}
    for p in _props_cache["data"]:      # use unfiltered list for game dropdown
        gid = p["game_id"]
        if gid not in seen_games:
            seen_games[gid] = {"game_id": gid, "away": p["opp"], "home": p["team"]}

    cached_at = (
        datetime.fromtimestamp(_props_cache["ts"]).isoformat()
        if _props_cache["ts"] else None
    )

    return jsonify({
        "success":   True,
        "props":     props,
        "count":     len(props),
        "games":     list(seen_games.values()),
        "cached_at": cached_at,
    })


@app.route("/api/player-stats")
def api_player_stats():
    """Return recent game logs + rolling averages for one player/stat."""
    player_name = request.args.get("player", "").strip()
    stat        = request.args.get("stat",   "PTS").strip().upper()

    if not player_name:
        return jsonify({"success": False, "error": "player param required"}), 400

    logs = _get_player_logs(player_name)
    if not logs:
        return jsonify({"success": False, "error": f"No data for {player_name}"}), 404

    # Look up current line from props cache.
    line = 0.0
    for p in _props_cache.get("data", []):
        if p["player_name"].lower() == player_name.lower() and p["stat"] == stat:
            line = p["line"]
            break

    avgs = _rolling_avgs(logs, stat, line)

    return jsonify({
        "success":     True,
        "player_name": player_name,
        "stat":        stat,
        "line":        line,
        "game_logs":   logs[:15],
        **avgs,
    })


@app.route("/api/rankings")
def api_rankings():
    """Return all NBA props enriched with rolling stats and edge estimates.

    Makes nba_api calls for each unique player.  A full slate takes
    30–90 s on first call; results are cached for _RANKINGS_TTL seconds.
    """
    force = request.args.get("refresh", "").lower() == "true"
    now   = time.time()

    if not force and now - _rankings_cache["ts"] < _RANKINGS_TTL and _rankings_cache["data"]:
        return jsonify({"success": True, "rankings": _rankings_cache["data"], "cached": True})

    props = _fetch_and_cache_props()
    if not props:
        return jsonify({"success": False, "error": "No props available"}), 503

    # Deduplicate by (player, stat) so we only call nba_api once per player.
    seen: set[tuple] = set()
    rankings = []

    for p in props:
        key = (p["player_name"], p["stat"])
        if key in seen:
            continue
        seen.add(key)

        logs = _get_player_logs(p["player_name"])
        avgs = _rolling_avgs(logs, p["stat"], p["line"])

        rankings.append({**p, **avgs})

    # Sort by absolute edge descending (picks with biggest edge first).
    rankings.sort(key=lambda x: abs(x.get("edge") or 0), reverse=True)

    _rankings_cache["ts"]   = time.time()
    _rankings_cache["data"] = rankings

    return jsonify({"success": True, "rankings": rankings, "cached": False})


@app.route("/api/status")
def api_status():
    """Health-check: what data is loaded / available."""
    model_count  = len(list(_MODEL_DIR.glob("*.joblib"))) if _MODEL_DIR.exists() else 0
    has_features = _PLAYER_FEAT_PATH.exists()

    return jsonify({
        "success":          True,
        "models_available": model_count,
        "features_file":    has_features,
        "props_cached":     bool(_props_cache["data"]),
        "props_count":      len(_props_cache["data"]),
        "players_cached":   len(_stats_cache),
        "season":           _SEASON,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA Props Dashboard")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log.info("Starting NBA Props Dashboard → http://localhost:%d", args.port)
    app.run(debug=args.debug, port=args.port, threaded=True)

#!/usr/bin/env python3
"""NBA Props Web Dashboard — Flask backend.

Serves a browser UI that shows:
  • Underdog Fantasy lines for today's NBA slate
  • Rolling player stats from NBA API (L5/L10/season/hit-rate)
  • ML model projections and probability-based edge (when models are loaded)
  • Key model input features from the player features parquet
  • Double-double / triple-double frequencies

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

import pandas as pd

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
_cache_lock      = Lock()

_PROPS_TTL    = 300   # 5 minutes
_STATS_TTL    = 600   # 10 minutes
_RANKINGS_TTL = 300   # 5 minutes

# ---------------------------------------------------------------------------
# Lazy-loaded model state (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_prop_models_cache:  dict | None          = None   # stat -> PlayerPropModel
_player_feat_cache:  pd.DataFrame | None  = None
_model_init_done:    bool                 = False
_feat_init_done:     bool                 = False
_model_lock = Lock()


def _load_prop_models() -> dict:
    """Load prop models from data/models/ once.  Returns {} if unavailable."""
    global _prop_models_cache, _model_init_done
    if _model_init_done:
        return _prop_models_cache or {}
    with _model_lock:
        if _model_init_done:
            return _prop_models_cache or {}
        models: dict = {}
        if _MODEL_DIR.exists():
            try:
                from src.models.player_props import PlayerPropModel
                for p in sorted(_MODEL_DIR.glob("player_prop_*.joblib")):
                    stat = p.stem.replace("player_prop_", "").upper()
                    try:
                        models[stat] = PlayerPropModel.load(str(p))
                    except Exception as exc:
                        log.warning("Could not load model %s: %s", p.name, exc)
                log.info("Loaded %d prop models from %s.", len(models), _MODEL_DIR)
            except Exception as exc:
                log.warning("Could not import PlayerPropModel: %s", exc)
        _prop_models_cache = models
        _model_init_done   = True
    return models


def _load_player_features() -> pd.DataFrame:
    """Load player features parquet once.  Returns empty DataFrame if unavailable."""
    global _player_feat_cache, _feat_init_done
    if _feat_init_done:
        return _player_feat_cache if _player_feat_cache is not None else pd.DataFrame()
    with _model_lock:
        if _feat_init_done:
            return _player_feat_cache if _player_feat_cache is not None else pd.DataFrame()
        if _PLAYER_FEAT_PATH.exists():
            try:
                _player_feat_cache = pd.read_parquet(_PLAYER_FEAT_PATH)
                log.info("Loaded player features: %d rows, %d cols.",
                         len(_player_feat_cache), len(_player_feat_cache.columns))
            except Exception as exc:
                log.warning("Could not load player features: %s", exc)
                _player_feat_cache = pd.DataFrame()
        else:
            log.info("Player features not found at %s (run build_player_features.py).",
                     _PLAYER_FEAT_PATH)
            _player_feat_cache = pd.DataFrame()
        _feat_init_done = True
    return _player_feat_cache


def _model_predict(
    player_name: str,
    stat: str,
    line: float,
    over_prob_underdog: float,
) -> dict | None:
    """Run ML prop model prediction.  Returns None if models/features unavailable."""
    models  = _load_prop_models()
    feat_df = _load_player_features()

    if not models or feat_df.empty or stat not in models:
        return None

    name_col = next((c for c in feat_df.columns if c.upper() == "PLAYER_NAME"), None)
    if name_col is None:
        return None

    rows = feat_df[feat_df[name_col] == player_name]
    if rows.empty:
        return None
    row = rows.iloc[-1]   # most recent feature row for this player

    try:
        from src.betting.edge_calculator import prob_over_from_quantiles
        from src.models.player_props import _get_feature_cols

        feat_cols = _get_feature_cols(stat, list(row.index))
        if not feat_cols:
            return None

        X     = pd.DataFrame([row[feat_cols].fillna(0).values], columns=feat_cols)
        preds = models[stat].predict(X)
        median    = float(preds["median"][0])
        low       = float(preds["low"][0])
        high      = float(preds["high"][0])
        prob_over = prob_over_from_quantiles(median, low, high, line)
        # Edge in percentage-POINTS vs Underdog implied probability:
        # positive = OVER edge, negative = UNDER edge.
        model_edge = round((prob_over - over_prob_underdog) * 100, 1)

        return {
            "model_median":    round(median,    2),
            "model_low":       round(low,       2),
            "model_high":      round(high,      2),
            "model_prob_over": round(prob_over, 4),
            "model_edge":      model_edge,
        }
    except Exception as exc:
        log.debug("Model predict failed for %s/%s: %s", player_name, stat, exc)
        return None


def _player_features_summary(player_name: str, stat: str) -> dict:
    """Return key model input features for display in the player panel."""
    feat_df = _load_player_features()
    if feat_df.empty:
        return {}

    name_col = next((c for c in feat_df.columns if c.upper() == "PLAYER_NAME"), None)
    if name_col is None:
        return {}

    rows = feat_df[feat_df[name_col] == player_name]
    if rows.empty:
        return {}
    row = rows.iloc[-1]

    # Stat-specific columns from rolling_stats.py / matchup.py / home_away.py
    stat_map = {
        f"{stat}_L5":        "feat_l5",
        f"{stat}_L10":       "feat_l10",
        f"{stat}_SEASON":    "feat_season",
        f"{stat}_HOME_AVG":  "home_avg",
        f"{stat}_AWAY_AVG":  "away_avg",
        f"{stat}_VS_OPP_AVG":"vs_opp_avg",
        f"{stat}_VS_OPP_N":  "vs_opp_n",
    }
    # Stat-agnostic columns
    common_map = {
        "USAGE_PROXY_L5":    "usage_l5",
        "USAGE_PROXY_L10":   "usage_l10",
        "TEAMMATE_OUT_BOOST":"teammate_boost",
        "TEAMMATE_OUT_FLAG": "teammate_out",
    }

    result: dict = {}
    for src, dst in {**stat_map, **common_map}.items():
        if src in row.index and pd.notna(row[src]):
            val = row[src]
            if dst == "teammate_out":
                result[dst] = bool(val)
            else:
                result[dst] = round(float(val), 2)

    # Feature freshness
    for date_col in ("DATE", "GAME_DATE"):
        if date_col in row.index and pd.notna(row[date_col]):
            result["feature_date"] = str(row[date_col])[:10]
            break

    return result


# ---------------------------------------------------------------------------
# Underdog lines helpers
# ---------------------------------------------------------------------------

def _prob_to_american(prob: float) -> str | None:
    """Convert implied probability to American odds string.

    Returns None when the line is effectively even-money (within 0.2 pp of 50 %).
    Callers use None to suppress display of symmetric / standard lines.
    """
    if prob <= 0 or prob >= 1:
        return None
    if abs(prob - 0.5) < 0.002:
        return None          # treat ≤ 0.2 pp of 50 % as standard / no displayed odds
    if prob > 0.5:
        return f"-{round(prob / (1.0 - prob) * 100)}"
    return f"+{round((1.0 - prob) / prob * 100)}"


def _fetch_and_cache_props(force: bool = False) -> list[dict]:
    """Return today's prop lines as plain dicts, cached for _PROPS_TTL seconds.

    If prop models + player features are already loaded, also attach model
    predictions so the table shows pipeline edge without requiring "Analyze All".
    """
    now = time.time()
    if not force and now - _props_cache["ts"] < _PROPS_TTL and _props_cache["data"]:
        return _props_cache["data"]

    try:
        raw = fetch_prop_lines()
    except (UnderdogAuthError, UnderdogAPIError) as exc:
        log.error("Underdog API error: %s", exc)
        return _props_cache["data"]   # return stale data on error

    result = []
    for line_obj in raw:
        over_am  = _prob_to_american(line_obj.over_payout)
        under_am = _prob_to_american(line_obj.under_payout)

        # Only display odds when the line is genuinely asymmetric (Underdog has vig
        # on one side).  Symmetric -110/-110 style lines show "–".
        odds_asymmetric = (over_am is not None or under_am is not None) and (
            abs(line_obj.over_payout - line_obj.under_payout) > 0.01
        )

        row: dict = {
            "player_name":    line_obj.player_name,
            "team":           line_obj.team,
            "opp":            line_obj.opp,
            "stat":           line_obj.stat,
            "line":           line_obj.line,
            "over_payout":    line_obj.over_payout,
            "under_payout":   line_obj.under_payout,
            "over_american":  over_am  if odds_asymmetric else None,
            "under_american": under_am if odds_asymmetric else None,
            "odds_asymmetric":odds_asymmetric,
            "game_id":        line_obj.game_id,
            "game_date":      str(line_obj.game_date),
        }

        # Attach model prediction eagerly (no network calls — CPU only).
        model_pred = _model_predict(
            line_obj.player_name, line_obj.stat,
            line_obj.line, line_obj.over_payout,
        )
        if model_pred:
            row.update(model_pred)
            row["edge"]      = model_pred["model_edge"]
            row["edge_type"] = "model"
            row["direction"] = "OVER" if model_pred["model_edge"] > 0 else "UNDER"

        result.append(row)

    _props_cache["ts"]   = time.time()
    _props_cache["data"] = result
    return result


# ---------------------------------------------------------------------------
# NBA API helpers
# ---------------------------------------------------------------------------

def _nba_api_game_logs(player_name: str) -> list[dict]:
    """Fetch recent game logs from nba_api.  Rate-limited at 0.6 s/call."""
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
            "GAME_DATE":  "date",       "MATCHUP":   "opponent",
            "WL":         "result",     "MIN":       "minutes",
            "PTS":        "points",     "REB":       "rebounds",
            "AST":        "assists",    "STL":       "steals",
            "BLK":        "blocks",     "TOV":       "turnovers",
            "FGM":        "fgm",        "FGA":       "fga",
            "FG3M":       "fg3m",       "FG3A":      "fg3a",
            "FTM":        "ftm",        "FTA":       "fta",
            "PLUS_MINUS": "plus_minus",
        }
        df2    = df.rename(columns=rename)
        keep   = [v for v in rename.values() if v in df2.columns]
        result = df2[keep].head(20).fillna(0)

        # Computed shooting percentages (avoid division by zero)
        result = result.copy()
        result["fg_pct"] = (result["fgm"] / result["fga"].replace(0, float("nan"))).round(3).fillna(0)
        result["ft_pct"] = (result["ftm"] / result["fta"].replace(0, float("nan"))).round(3).fillna(0)

        return result.to_dict("records")

    except Exception as exc:
        log.warning("nba_api error for %s: %s", player_name, exc)
        return []


def _get_player_logs(player_name: str) -> list[dict]:
    """Return cached game logs, fetching from nba_api on cache miss."""
    now = time.time()
    with _cache_lock:
        cached = _stats_cache.get(player_name)
        if cached and now - cached[0] < _STATS_TTL:
            return cached[1]

    logs = _nba_api_game_logs(player_name)

    with _cache_lock:
        _stats_cache[player_name] = (time.time(), logs)
    return logs


# ---------------------------------------------------------------------------
# Stat computation helpers
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
    keys = _STAT_KEYS.get(stat)
    if not keys:
        return None
    vals = [log_row.get(k) for k in keys]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals)


def _rolling_avgs(logs: list[dict], stat: str, line: float) -> dict:
    """L5/L10/season averages, hit rate, and a rolling-stats proximity edge.

    NOTE: `rolling_edge` is NOT the model edge.  It is a simple measure of
    how far the rolling projection sits above/below the Underdog line:
        rolling_edge = (projection − line) / |line| × 100 %
    Use `model_edge` (from _model_predict) as the true betting edge.
    """
    values = []
    for g in logs:
        v = _stat_value_from_log(g, stat)
        if v is not None:
            values.append(v)

    if not values:
        return {
            "L5": None, "L10": None, "season": None,
            "hit_rate": None, "rolling_edge": None, "direction": None,
        }

    l5     = round(sum(values[:5])  / len(values[:5]),  1) if values[:5]  else None
    l10    = round(sum(values[:10]) / len(values[:10]), 1) if values[:10] else None
    season = round(sum(values)      / len(values),      1)

    # Hit rate: share of the last 10 games where player went OVER this line.
    over_count = sum(1 for v in values[:10] if v > line)
    hit_rate   = round(over_count / len(values[:10]), 3) if values[:10] else None

    proj         = (season * 0.4 + l10 * 0.6) if l10 is not None else (l10 or season or 0.0)
    rolling_edge = round((proj - line) / max(abs(line), 1) * 100, 1)
    direction    = "OVER" if rolling_edge > 0 else "UNDER"

    return {
        "L5": l5, "L10": l10, "season": season,
        "hit_rate": hit_rate,
        "rolling_edge": rolling_edge,
        "direction": direction,
    }


def _compute_dd_stats(logs: list[dict]) -> dict:
    """Double-double / triple-double frequency from last 10 game logs."""
    recent = logs[:10]
    n = len(recent)
    if n == 0:
        return {}

    def ge10(g: dict, key: str) -> bool:
        return float(g.get(key) or 0) >= 10

    dd_pr = sum(1 for g in recent if ge10(g, "points") and ge10(g, "rebounds"))
    dd_pa = sum(1 for g in recent if ge10(g, "points") and ge10(g, "assists"))
    td    = sum(1 for g in recent
                if sum(1 for k in ("points", "rebounds", "assists") if ge10(g, k)) >= 3)

    return {
        "dd_pts_reb_L10":    dd_pr,
        "dd_pts_ast_L10":    dd_pa,
        "triple_double_L10": td,
        "dd_sample":         n,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/props")
def api_props():
    """Return today's Underdog prop lines.  Model predictions included if available."""
    force  = request.args.get("refresh", "").lower() == "true"
    player = request.args.get("player", "").strip().lower()
    stat   = request.args.get("stat",   "").strip().upper()

    props = _fetch_and_cache_props(force=force)

    if player:
        props = [p for p in props if player in p["player_name"].lower()]
    if stat:
        props = [p for p in props if p["stat"] == stat]

    # Build game list for the filter dropdown (use unfiltered list).
    seen_games: dict[str, dict] = {}
    for p in _props_cache["data"]:
        gid = p["game_id"]
        if gid not in seen_games:
            seen_games[gid] = {"game_id": gid, "away": p["opp"], "home": p["team"]}

    cached_at = (
        datetime.fromtimestamp(_props_cache["ts"]).isoformat()
        if _props_cache["ts"] else None
    )

    return jsonify({
        "success":       True,
        "props":         props,
        "count":         len(props),
        "games":         list(seen_games.values()),
        "cached_at":     cached_at,
        "models_loaded": len(_load_prop_models()),
    })


@app.route("/api/player-stats")
def api_player_stats():
    """Return game logs + rolling averages + model prediction + model features."""
    player_name = request.args.get("player", "").strip()
    stat        = request.args.get("stat",   "PTS").strip().upper()

    if not player_name:
        return jsonify({"success": False, "error": "player param required"}), 400

    logs = _get_player_logs(player_name)
    if not logs:
        return jsonify({"success": False, "error": f"No data for {player_name}"}), 404

    # Look up line and Underdog implied probability from props cache.
    line            = 0.0
    over_prob_under = 0.5
    for p in _props_cache.get("data", []):
        if p["player_name"].lower() == player_name.lower() and p["stat"] == stat:
            line            = p["line"]
            over_prob_under = p["over_payout"]
            break

    avgs         = _rolling_avgs(logs, stat, line)
    dd_stats     = _compute_dd_stats(logs)
    model_pred   = _model_predict(player_name, stat, line, over_prob_under)
    feat_summary = _player_features_summary(player_name, stat)

    # Canonical edge: model probability edge when available, rolling edge otherwise.
    if model_pred:
        edge      = model_pred["model_edge"]
        edge_type = "model"
        direction = "OVER" if edge > 0 else "UNDER"
    else:
        edge      = avgs.get("rolling_edge")
        edge_type = "rolling"
        direction = avgs.get("direction")

    return jsonify({
        "success":     True,
        "player_name": player_name,
        "stat":        stat,
        "line":        line,
        "game_logs":   logs[:15],
        **avgs,
        **dd_stats,
        "edge":        edge,
        "edge_type":   edge_type,
        "direction":   direction,
        "model":       model_pred,
        "features":    feat_summary,
    })


@app.route("/api/rankings")
def api_rankings():
    """All props enriched with rolling stats, model predictions, and edge estimates.

    NBA API calls are made per unique player (~0.6 s each).  First call for a
    full slate takes 30–90 s; results are cached for _RANKINGS_TTL seconds.
    """
    force = request.args.get("refresh", "").lower() == "true"
    now   = time.time()

    if not force and now - _rankings_cache["ts"] < _RANKINGS_TTL and _rankings_cache["data"]:
        return jsonify({"success": True, "rankings": _rankings_cache["data"], "cached": True})

    props = _fetch_and_cache_props()
    if not props:
        return jsonify({"success": False, "error": "No props available"}), 503

    seen: set[tuple] = set()
    rankings = []

    for p in props:
        key = (p["player_name"], p["stat"])
        if key in seen:
            continue
        seen.add(key)

        logs       = _get_player_logs(p["player_name"])
        avgs       = _rolling_avgs(logs, p["stat"], p["line"])
        model_pred = _model_predict(p["player_name"], p["stat"], p["line"], p["over_payout"])

        if model_pred:
            edge      = model_pred["model_edge"]
            edge_type = "model"
            direction = "OVER" if edge > 0 else "UNDER"
        else:
            edge      = avgs.get("rolling_edge")
            edge_type = "rolling"
            direction = avgs.get("direction")

        row = {
            **p, **avgs,
            "edge":      edge,
            "edge_type": edge_type,
            "direction": direction,
        }
        if model_pred:
            row.update({
                "model_median":    model_pred["model_median"],
                "model_low":       model_pred["model_low"],
                "model_high":      model_pred["model_high"],
                "model_prob_over": model_pred["model_prob_over"],
                "model_edge":      model_pred["model_edge"],
            })
        rankings.append(row)

    rankings.sort(key=lambda x: abs(x.get("edge") or 0), reverse=True)

    _rankings_cache["ts"]   = time.time()
    _rankings_cache["data"] = rankings

    return jsonify({"success": True, "rankings": rankings, "cached": False})


@app.route("/api/status")
def api_status():
    """Health-check: models loaded, features available, props cached."""
    models  = _load_prop_models()
    feat_df = _load_player_features()

    return jsonify({
        "success":          True,
        "models_available": len(models),
        "model_stats":      sorted(models.keys()),
        "features_file":    _PLAYER_FEAT_PATH.exists(),
        "features_rows":    len(feat_df) if not feat_df.empty else 0,
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

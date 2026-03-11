"""Underdog Fantasy public API client.

Fetches today's player prop lines (over/under) from Underdog's
public REST API.  No authentication or token required.

The endpoint /beta/v6/over_under_lines returns all currently active
pick'em lines across all sports.  We filter for sport_id == "NBA".

Usage:
    from src.data.scrapers.underdog import fetch_prop_lines, save_lines_to_db
    props = fetch_prop_lines()
    save_lines_to_db(props, [])
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests
from dotenv import load_dotenv

from src.data.db import get_cursor

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

_API_BASE        = "https://api.underdogfantasy.com"
_TIMEOUT         = 15
_PUBLIC_ENDPOINT = "/beta/v6/over_under_lines"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# Underdog internal stat name -> our canonical convention.
STAT_MAP: dict[str, str] = {
    "points":               "PTS",
    "rebounds":             "REB",
    "assists":              "AST",
    "pts_rebs_asts":        "PRA",
    "pts_rebs":             "PR",
    "pts_asts":             "PA",
    "rebs_asts":            "RA",
    "three_pointers_made":  "FG3M",
    "3pt_fg_made":          "FG3M",
    "steals":               "STL",
    "blocks":               "BLK",
    "turnovers":            "TOV",
    "fantasy_points":       "FAN",
    "minutes":              "MIN",
    # Legacy / alternate spellings observed in the wild:
    "pts_reb_ast":          "PRA",
    "reb_ast":              "RA",
    "three_pt_fg":          "FG3M",
    "fg3m":                 "FG3M",
    "tov":                  "TOV",
    "stl":                  "STL",
    "blk":                  "BLK",
    "min":                  "MIN",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnderdogAuthError(Exception):
    """Raised on 401 (endpoint may require auth in the future)."""


class UnderdogAPIError(Exception):
    """Raised for unexpected non-200 API responses."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UnderdogPropLine:
    player_id:    str
    player_name:  str
    team:         str
    opp:          str
    game_id:      str
    stat:         str        # canonical (PTS, REB, PRA, ...)
    line:         float
    over_payout:  float      # implied probability for OVER (from american_price, or 0.5)
    under_payout: float      # implied probability for UNDER (from american_price, or 0.5)
    game_date:    date


@dataclass
class UnderdogGameLine:
    """Placeholder -- Rival (game-winner) lines not yet implemented."""
    game_id:     str
    home_team:   str
    away_team:   str
    home_payout: float
    away_payout: float
    game_date:   date


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_public(path: str) -> dict:
    """Unauthenticated GET to the Underdog public API."""
    url  = f"{_API_BASE}{path}"
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)

    if resp.status_code == 401:
        raise UnderdogAuthError(
            f"401 Unauthorized from {url}. "
            "The endpoint may now require authentication."
        )
    if resp.status_code != 200:
        raise UnderdogAPIError(
            f"Underdog API returned {resp.status_code} for {url}: {resp.text[:200]}"
        )
    return resp.json()


def _normalize_stat(raw: str) -> str:
    key = raw.lower().strip()
    return STAT_MAP.get(key, raw.upper())


def _american_to_prob(american_price: str) -> float:
    """Convert American odds string (e.g. '-125', '+102') to implied probability."""
    odds = int(american_price)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


# ---------------------------------------------------------------------------
# Prop lines
# ---------------------------------------------------------------------------

def fetch_prop_lines(date_str: str | None = None) -> list[UnderdogPropLine]:
    """Fetch today's NBA player prop over/under lines (no auth required).

    The public endpoint returns all active lines across all sports.
    This function filters to sport_id == "NBA" only.

    Args:
        date_str: ISO date string (YYYY-MM-DD) used as fallback when
                  game scheduled_at is unparseable.  Defaults to today.

    Returns:
        List of UnderdogPropLine objects (one per player/stat/line).

    Raises:
        UnderdogAuthError: Endpoint returned 401.
        UnderdogAPIError:  Non-200 API response.
    """
    target_date = date.fromisoformat(date_str) if date_str else date.today()
    log.info("Fetching Underdog prop lines (public endpoint) ...")

    data = _get_public(_PUBLIC_ENDPOINT)

    # Public endpoint returns lists; convert to dicts for O(1) lookup.
    appearances_list = data.get("appearances", [])
    games_list       = data.get("games", [])
    ou_lines_list    = data.get("over_under_lines", [])

    appearances = {a["id"]: a for a in appearances_list}
    games       = {g["id"]: g for g in games_list}

    # Only process NBA games.
    nba_game_ids = {g["id"] for g in games_list if g.get("sport_id") == "NBA"}

    # Build team_id -> abbreviation from game titles "AWAY @ HOME".
    team_id_to_abbr: dict[str, str] = {}
    game_id_to_info: dict = {}

    for g in games_list:
        gid = g.get("id")
        if gid not in nba_game_ids:
            continue
        # Support both "title" (public endpoint) and "abbreviated_title" (auth endpoint).
        title = g.get("title", "") or g.get("abbreviated_title", "")
        parts = title.split(" @ ")
        if len(parts) != 2:
            continue
        away_abbr, home_abbr = parts[0].strip(), parts[1].strip()
        home_tid = g.get("home_team_id", "")
        away_tid = g.get("away_team_id", "")
        team_id_to_abbr[home_tid] = home_abbr
        team_id_to_abbr[away_tid] = away_abbr
        game_id_to_info[gid] = {
            "home_abbr":    home_abbr,
            "away_abbr":    away_abbr,
            "home_tid":     home_tid,
            "away_tid":     away_tid,
            "scheduled_at": g.get("scheduled_at", ""),
        }

    lines: list[UnderdogPropLine] = []

    for ou_line in ou_lines_list:
        try:
            if ou_line.get("status") != "active":
                continue

            stat_value = float(ou_line.get("stat_value", 0))
            ou         = ou_line.get("over_under", {})
            app_stat   = ou.get("appearance_stat", {})
            stat       = _normalize_stat(app_stat.get("stat", ""))
            if not stat:
                continue

            appearance_id = app_stat.get("appearance_id", "")
            appearance    = appearances.get(appearance_id, {})
            team_id       = appearance.get("team_id", "")
            match_id      = appearance.get("match_id")

            if match_id not in nba_game_ids:
                continue

            team_abbr = team_id_to_abbr.get(team_id, "???")
            game_info = game_id_to_info.get(match_id, {})
            opp_abbr  = (
                game_info.get("away_abbr", "???")
                if team_id == game_info.get("home_tid")
                else game_info.get("home_abbr", "???")
            )

            player_name:    str        = ""
            over_american:  str | None = None
            under_american: str | None = None

            for opt in ou_line.get("options", []):
                if opt.get("choice") == "higher":
                    player_name   = opt.get("selection_header", player_name)
                    over_american = opt.get("american_price")
                elif opt.get("choice") == "lower":
                    if not player_name:
                        player_name = opt.get("selection_header", "")
                    under_american = opt.get("american_price")

            if not player_name:
                continue

            # american_price may be absent on the public endpoint — default to 0.5.
            over_prob  = _american_to_prob(over_american)  if over_american  else 0.5
            under_prob = _american_to_prob(under_american) if under_american else 0.5

            scheduled = game_info.get("scheduled_at", "")
            try:
                game_date = datetime.fromisoformat(
                    scheduled.replace("Z", "+00:00")
                ).astimezone(timezone.utc).date()
            except (ValueError, AttributeError):
                game_date = target_date

            lines.append(UnderdogPropLine(
                player_id    = appearance_id,
                player_name  = player_name.strip(),
                team         = team_abbr,
                opp          = opp_abbr,
                game_id      = str(match_id),
                stat         = stat,
                line         = stat_value,
                over_payout  = round(over_prob,  4),
                under_payout = round(under_prob, 4),
                game_date    = game_date,
            ))

        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping malformed line: %s", exc)

    log.info("Parsed %d NBA prop lines.", len(lines))
    return lines


def fetch_game_lines(date_str: str | None = None) -> list[UnderdogGameLine]:
    """Fetch game-winner (Rival) lines.

    NOTE: The Rival endpoint has not been observed in network traffic yet.
    Returns an empty list until the endpoint is identified.
    """
    log.warning(
        "fetch_game_lines: Rival endpoint not yet implemented. "
        "Returning empty list."
    )
    return []


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------

def save_lines_to_db(
    prop_lines: list[UnderdogPropLine],
    game_lines: list[UnderdogGameLine],
) -> int:
    """Upsert fetched lines into the underdog_lines table.

    Returns:
        Number of rows written.
    """
    rows_saved = 0
    with get_cursor() as cur:
        for pl in prop_lines:
            cur.execute(
                """
                INSERT INTO underdog_lines
                    (stat_type, player_id, player_name, team, opp, game_id,
                     stat, line, over_payout, under_payout, game_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id, COALESCE(player_id, ''), stat, game_date)
                DO UPDATE SET
                    line         = EXCLUDED.line,
                    over_payout  = EXCLUDED.over_payout,
                    under_payout = EXCLUDED.under_payout,
                    fetched_at   = NOW()
                """,
                (
                    "OVER_UNDER",
                    pl.player_id, pl.player_name, pl.team, pl.opp,
                    pl.game_id, pl.stat, pl.line,
                    pl.over_payout, pl.under_payout, pl.game_date,
                ),
            )
            rows_saved += cur.rowcount

        for gl in game_lines:
            cur.execute(
                """
                INSERT INTO underdog_lines
                    (stat_type, player_id, player_name, team, opp, game_id,
                     stat, line, over_payout, under_payout, game_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id, COALESCE(player_id, ''), stat, game_date)
                DO UPDATE SET
                    over_payout  = EXCLUDED.over_payout,
                    under_payout = EXCLUDED.under_payout,
                    fetched_at   = NOW()
                """,
                (
                    "RIVAL",
                    None, None, gl.home_team, gl.away_team,
                    gl.game_id, "GAME", None,
                    gl.home_payout, gl.away_payout, gl.game_date,
                ),
            )
            rows_saved += cur.rowcount

    log.info("Saved %d Underdog line rows to DB.", rows_saved)
    return rows_saved

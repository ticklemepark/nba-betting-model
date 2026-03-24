"""NBA injury report scraper.

Uses the ESPN public JSON API (no auth required, updates ~30 min after NBA release):
    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

Returns a clean DataFrame with one row per player listed on the report.

The HTML scraper (parse_injury_report) is retained for unit tests that pass
fixture HTML.  All production fetches should use fetch_injury_report() which
calls the JSON endpoint.

Usage:
    from src.data.scrapers.injury_report import fetch_injury_report

    df = fetch_injury_report()
    # Columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT
    # TEAM is a 3-letter nba_api abbreviation (e.g. "LAL", "BKN", "CHA")

    # Filter to players who are definitely out:
    out = df[df["STATUS"] == "OUT"]

    # Or parse from fixture JSON (for testing):
    df = parse_injury_json(api_response_dict)
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.constants import BBREF_TO_NBA_API, TEAM_NAMES

log = logging.getLogger(__name__)

_ESPN_INJURY_URL = "https://www.espn.com/nba/injuries"
_ESPN_INJURY_JSON_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_RATE_LIMIT_SECONDS = 1.0

# ESPN status strings → normalised status
_STATUS_MAP = {
    "out":          "OUT",
    "doubtful":     "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "probable":     "PROBABLE",
    "day-to-day":   "QUESTIONABLE",
    "day to day":   "QUESTIONABLE",
}

EXPECTED_COLUMNS = ["PLAYER_NAME", "TEAM", "STATUS", "REASON", "FETCHED_AT"]

# ESPN full team name → nba_api 3-letter abbreviation.
# Built from TEAM_NAMES (bbref abbr → name), applying the 3 bbref→nba_api fixes.
ESPN_NAME_TO_ABBR: dict[str, str] = {
    name: BBREF_TO_NBA_API.get(abbr, abbr)
    for abbr, name in TEAM_NAMES.items()
}


# ---------------------------------------------------------------------------
# Public API — JSON endpoint (production)
# ---------------------------------------------------------------------------

def fetch_injury_report(session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch the current NBA injury report from the ESPN JSON API.

    Args:
        session: Optional requests.Session for connection reuse / testing.

    Returns:
        DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT.
        TEAM is a 3-letter nba_api abbreviation (LAL, BKN, CHA, PHX, etc.).
        STATUS is one of: OUT / DOUBTFUL / QUESTIONABLE / PROBABLE.
        FETCHED_AT is a UTC timestamp string.

        Returns an empty DataFrame on network or parse failure.
    """
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; nba-betting-model/1.0)"})

    try:
        resp = sess.get(_ESPN_INJURY_JSON_URL, timeout=15)
        resp.raise_for_status()
        time.sleep(_RATE_LIMIT_SECONDS)
        data = resp.json()
    except requests.RequestException as exc:
        log.error("Failed to fetch ESPN injury JSON: %s", exc)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    except ValueError as exc:
        log.error("ESPN injury API returned non-JSON: %s", exc)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    return parse_injury_json(data)


def parse_injury_json(data: dict) -> pd.DataFrame:
    """Parse ESPN injury API JSON response into a clean DataFrame.

    Args:
        data: Parsed JSON from the ESPN injuries endpoint.
              Expected shape: {"injuries": [{"displayName": "Atlanta Hawks",
                                             "injuries": [{"athlete": {...},
                                                           "status": "...",
                                                           "shortComment": "..."},
                                                          ...]}, ...]}

    Returns:
        DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT.
    """
    rows = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    team_entries = data.get("injuries", [])
    for team_entry in team_entries:
        team_display = team_entry.get("displayName", "")
        team_abbr = ESPN_NAME_TO_ABBR.get(team_display, team_display[:3].upper())

        for injury in team_entry.get("injuries", []):
            athlete = injury.get("athlete", {})
            player_name = athlete.get("displayName", "")
            if not player_name:
                player_name = athlete.get("fullName", "")

            status_raw = injury.get("status", "").lower()
            status = _STATUS_MAP.get(status_raw, status_raw.upper() if status_raw else "UNKNOWN")

            reason = injury.get("shortComment", "") or injury.get("longComment", "")

            if player_name:
                rows.append({
                    "PLAYER_NAME": player_name,
                    "TEAM":        team_abbr,
                    "STATUS":      status,
                    "REASON":      reason,
                    "FETCHED_AT":  fetched_at,
                })

    if not rows:
        log.warning("No injury entries found in ESPN JSON response.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(rows)[EXPECTED_COLUMNS]
    log.info("Parsed %d injury entries from ESPN JSON API.", len(df))
    return df


# ---------------------------------------------------------------------------
# HTML-based parser — retained for unit tests using fixture HTML
# ---------------------------------------------------------------------------

def parse_injury_report(html: str) -> pd.DataFrame:
    """Parse ESPN injury page HTML into a clean DataFrame.

    NOTE: ESPN now renders injury tables via JavaScript so this function will
    return an empty DataFrame on real ESPN pages.  It is retained to support
    existing unit tests that pass fixture HTML strings.

    For production use, call fetch_injury_report() which uses the JSON API.

    Args:
        html: Full HTML of https://www.espn.com/nba/injuries.

    Returns:
        DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for table_wrapper in soup.select("div.ResponsiveTable"):
        team_elem = table_wrapper.find_previous(["h2", "h3", "div"], class_=lambda c: c and "TeamName" in c)
        team_name = team_elem.get_text(strip=True) if team_elem else "UNKNOWN"

        table = table_wrapper.find("table")
        if not table:
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 3:
                continue

            player_name = cells[0].get_text(strip=True)
            status_raw  = cells[1].get_text(strip=True).lower()
            reason      = cells[2].get_text(strip=True) if len(cells) > 2 else ""

            status = _STATUS_MAP.get(status_raw, status_raw.upper())

            if player_name:
                rows.append({
                    "PLAYER_NAME": player_name,
                    "TEAM":        team_name,
                    "STATUS":      status,
                    "REASON":      reason,
                    "FETCHED_AT":  fetched_at,
                })

    if not rows:
        log.warning("No injury rows parsed from ESPN injury report — HTML structure may have changed.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(rows)[EXPECTED_COLUMNS]
    log.info("Parsed %d injury entries from ESPN.", len(df))
    return df

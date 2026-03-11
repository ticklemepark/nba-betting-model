"""NBA injury report scraper.

Fetches the current NBA injury report from ESPN's injury page
(https://www.espn.com/nba/injuries).  Returns a clean DataFrame with
one row per player listed on the report.

Why ESPN: The NBA official injury report is a PDF requiring special parsing.
ESPN's HTML page is stable, well-structured, and updates within ~30 minutes
of the NBA's official release.

Rate limiting: a single page fetch per call (no multi-page pagination needed).

Usage:
    from src.data.scrapers.injury_report import fetch_injury_report, parse_injury_report

    df = fetch_injury_report()
    # Columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT

    # Or parse from HTML string (for testing):
    df = parse_injury_report(html)
"""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

_ESPN_INJURY_URL = "https://www.espn.com/nba/injuries"
_RATE_LIMIT_SECONDS = 2.0

# ESPN status strings → normalised status
_STATUS_MAP = {
    "out":          "OUT",
    "doubtful":     "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "probable":     "PROBABLE",
    "day-to-day":   "QUESTIONABLE",
}

EXPECTED_COLUMNS = ["PLAYER_NAME", "TEAM", "STATUS", "REASON", "FETCHED_AT"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_injury_report(session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch and parse the current NBA injury report from ESPN.

    Args:
        session: Optional requests.Session for connection reuse / testing.

    Returns:
        DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT.
        STATUS is normalised to one of: OUT / DOUBTFUL / QUESTIONABLE / PROBABLE.
        FETCHED_AT is a UTC timestamp string.

        Returns an empty DataFrame on network or parse failure.
    """
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; nba-betting-model/1.0)"})

    try:
        resp = sess.get(_ESPN_INJURY_URL, timeout=15)
        resp.raise_for_status()
        time.sleep(_RATE_LIMIT_SECONDS)
    except requests.RequestException as exc:
        log.error("Failed to fetch ESPN injury report: %s", exc)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    return parse_injury_report(resp.text)


def parse_injury_report(html: str) -> pd.DataFrame:
    """Parse ESPN injury page HTML into a clean DataFrame.

    Args:
        html: Full HTML of https://www.espn.com/nba/injuries.

    Returns:
        DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON, FETCHED_AT.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ESPN structures each team's injuries inside a div.ResponsiveTable
    for table_wrapper in soup.select("div.ResponsiveTable"):
        # Team name is in the preceding headline or title element
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

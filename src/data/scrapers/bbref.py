"""Basketball Reference team-level box score scraper.

Scrapes completed NBA game box scores from basketball-reference.com and
returns them as a pandas DataFrame whose column schema matches the original
feature-engineering notebook exactly.

Output columns (43 total):
    AWAY_MP  AWAY_FG  AWAY_FGA  AWAY_FG%  AWAY_3P  AWAY_3PA  AWAY_3P%
    AWAY_FT  AWAY_FTA AWAY_FT%  AWAY_ORB  AWAY_DRB  AWAY_TRB
    AWAY_AST AWAY_STL AWAY_BLK  AWAY_TO   AWAY_PF   AWAY_PTS
    (same 19 for HOME_)
    AWAY  HOME  DATE  SEASON

Rate limiting: 3-second sleep after every HTTP request.  Basketball
Reference's robots.txt does not prohibit scraping but the site will
throttle/block aggressive bots.  Do not reduce RATE_LIMIT_SECONDS.

Usage:
    from src.data.scrapers.bbref import scrape_season, scrape_seasons

    df_2024 = scrape_season(2024)          # 2023-24 regular season
    df_all  = scrape_seasons([2022, 2023, 2024])
"""

import logging
import re
import time
from typing import Generator

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.constants import BBREF_TEAMS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBREF_BASE = "https://www.basketball-reference.com"
RATE_LIMIT_SECONDS = 3.0

# Months that appear in an NBA regular season schedule page.
# Basketball Reference uses the season END year in the URL, so
# "NBA_2024_games-october.html" covers October 2023 (start of 2023-24).
NBA_REGULAR_SEASON_MONTHS: list[str] = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
]

# Maps Basketball Reference <td data-stat="..."> attributes to our column names.
# Order is preserved (Python 3.7+) and defines the stat column ordering.
_BBREF_STAT_TO_COL: dict[str, str] = {
    "mp":       "MP",
    "fg":       "FG",
    "fga":      "FGA",
    "fg_pct":   "FG%",
    "fg3":      "3P",
    "fg3a":     "3PA",
    "fg3_pct":  "3P%",
    "ft":       "FT",
    "fta":      "FTA",
    "ft_pct":   "FT%",
    "orb":      "ORB",
    "drb":      "DRB",
    "trb":      "TRB",
    "ast":      "AST",
    "stl":      "STL",
    "blk":      "BLK",
    "tov":      "TO",   # renamed: bbref uses "tov", notebook uses "TO"
    "pf":       "PF",
    "pts":      "PTS",
}

# The full ordered list of output columns (matches notebook schema).
EXPECTED_COLUMNS: list[str] = [
    "AWAY_MP",  "AWAY_FG",  "AWAY_FGA", "AWAY_FG%",
    "AWAY_3P",  "AWAY_3PA", "AWAY_3P%",
    "AWAY_FT",  "AWAY_FTA", "AWAY_FT%",
    "AWAY_ORB", "AWAY_DRB", "AWAY_TRB",
    "AWAY_AST", "AWAY_STL", "AWAY_BLK", "AWAY_TO", "AWAY_PF", "AWAY_PTS",
    "HOME_MP",  "HOME_FG",  "HOME_FGA", "HOME_FG%",
    "HOME_3P",  "HOME_3PA", "HOME_3P%",
    "HOME_FT",  "HOME_FTA", "HOME_FT%",
    "HOME_ORB", "HOME_DRB", "HOME_TRB",
    "HOME_AST", "HOME_STL", "HOME_BLK", "HOME_TO", "HOME_PF", "HOME_PTS",
    "AWAY", "HOME", "DATE", "SEASON",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_season(
    season: int,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Scrape all regular-season box scores for a given season year.

    Args:
        season: The season END year (e.g. 2024 for the 2023-24 season).
        session: Optional Session for connection reuse.  A new session with
                 a browser-like User-Agent is created if None.

    Returns:
        DataFrame with EXPECTED_COLUMNS.  May be empty if Basketball Reference
        returned no data (season not started, page structure changed, etc.).
    """
    if session is None:
        session = _make_session()

    records: list[dict] = []
    for month in NBA_REGULAR_SEASON_MONTHS:
        urls = _fetch_game_urls_for_month(season, month, session)
        log.info("Season %d | %s → %d game(s)", season, month, len(urls))
        for url in urls:
            record = _fetch_and_parse_box_score(url, season, session)
            if record is not None:
                records.append(record)

    if not records:
        log.warning("No box scores collected for season %d", season)
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    df = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
    _validate(df, season)
    return df


def scrape_seasons(seasons: list[int]) -> pd.DataFrame:
    """Scrape multiple seasons and return a single sorted DataFrame.

    Args:
        seasons: List of season END years, e.g. [2022, 2023, 2024].

    Returns:
        Concatenated DataFrame sorted by DATE, reset index.
    """
    session = _make_session()
    frames = [scrape_season(s, session) for s in seasons]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    result = pd.concat(non_empty, ignore_index=True)
    return result.sort_values("DATE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Schedule page: fetch game URLs for one month
# ---------------------------------------------------------------------------

def _fetch_game_urls_for_month(
    season: int,
    month: str,
    session: requests.Session,
) -> list[str]:
    """Return completed box score URLs from one monthly schedule page.

    Returns an empty list on 404 (month doesn't exist for this season)
    or any network error — the caller logs and moves on.
    """
    url = f"{BBREF_BASE}/leagues/NBA_{season}_games-{month}.html"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return []   # normal: e.g. October page for a season that started in Nov
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch schedule %s: %s", url, exc)
        return []
    finally:
        time.sleep(RATE_LIMIT_SECONDS)

    return _parse_game_urls(resp.text)


def _parse_game_urls(html: str) -> list[str]:
    """Extract completed game box score URLs from a schedule page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="schedule")
    if table is None:
        log.warning("No #schedule table found — page structure may have changed")
        return []

    tbody = table.find("tbody")
    if tbody is None:
        return []

    urls: list[str] = []
    for row in tbody.find_all("tr"):
        # Basketball Reference inserts month-separator rows with class "thead"
        if "thead" in (row.get("class") or []):
            continue

        link_cell = row.find("td", {"data-stat": "box_score_text"})
        if link_cell is None:
            continue

        anchor = link_cell.find("a")
        if anchor is None:
            continue    # future game — no box score link yet

        href = anchor.get("href", "")
        if href:
            urls.append(BBREF_BASE + href)

    return urls


# ---------------------------------------------------------------------------
# Box score page: fetch + parse one game
# ---------------------------------------------------------------------------

def _fetch_and_parse_box_score(
    url: str,
    season: int,
    session: requests.Session,
) -> dict | None:
    """Fetch a game page and return a parsed row dict, or None on any failure."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Could not fetch box score %s: %s", url, exc)
        return None
    finally:
        time.sleep(RATE_LIMIT_SECONDS)

    try:
        return _parse_box_score(resp.text, url, season)
    except (ValueError, AttributeError, IndexError) as exc:
        log.warning("Could not parse box score %s: %s", url, exc)
        return None


def _parse_box_score(html: str, url: str, season: int) -> dict:
    """Parse team-level totals from a game page HTML string.

    Args:
        html:   Raw HTML of the Basketball Reference game page.
        url:    Original URL (used for date extraction and error context).
        season: Season END year, e.g. 2024.

    Returns:
        Dict with exactly the keys in EXPECTED_COLUMNS.

    Raises:
        ValueError: If required page elements are missing or malformed.
    """
    soup = BeautifulSoup(html, "html.parser")

    away_abbr, home_abbr = _extract_team_abbrs(soup, url)
    date_str = _extract_date_from_url(url)

    away_stats = _parse_team_totals(soup, away_abbr)
    home_stats = _parse_team_totals(soup, home_abbr)

    row: dict = {}
    for col, val in away_stats.items():
        row[f"AWAY_{col}"] = val
    for col, val in home_stats.items():
        row[f"HOME_{col}"] = val
    row["AWAY"]   = away_abbr
    row["HOME"]   = home_abbr
    row["DATE"]   = date_str
    row["SEASON"] = season
    return row


def _extract_team_abbrs(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    """Return (away_abbr, home_abbr) by reading the scorebox.

    Basketball Reference lists the visiting team first and the home team
    second in the scorebox.  Each has an anchor tag with an href of the
    form /teams/{ABBR}/{YEAR}.html.

    Raises:
        ValueError: scorebox missing or fewer than 2 distinct team links.
    """
    scorebox = soup.find("div", class_="scorebox")
    if scorebox is None:
        raise ValueError(f"No .scorebox div found: {url}")

    team_href_pat = re.compile(r"/teams/([A-Z]{2,3})/\d{4}\.html")
    seen: list[str] = []
    for anchor in scorebox.find_all("a", href=team_href_pat):
        abbr = team_href_pat.search(anchor["href"]).group(1)
        if abbr not in seen:
            seen.append(abbr)
        if len(seen) == 2:
            break

    if len(seen) < 2:
        raise ValueError(
            f"Expected 2 team abbreviations in scorebox, found {seen}: {url}"
        )

    return seen[0], seen[1]   # (away, home)


def _extract_date_from_url(url: str) -> str:
    """Extract the game date as YYYY-MM-DD from a Basketball Reference URL.

    URL format: .../boxscores/YYYYMMDD{N}{TEAM}.html
    where N is a single-digit game number (usually 0).
    """
    match = re.search(r"/boxscores/(\d{4})(\d{2})(\d{2})\d", url)
    if not match:
        raise ValueError(f"Cannot parse date from URL: {url}")
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _parse_team_totals(soup: BeautifulSoup, team_abbr: str) -> dict[str, str | None]:
    """Parse the <tfoot> totals row from a team's basic box score table.

    The table id is box-{TEAM_ABBR}-game-basic.

    Args:
        soup:      Parsed page.
        team_abbr: Basketball Reference team abbreviation (e.g. "LAL").

    Returns:
        Dict mapping our column name (e.g. "PTS") to the raw string from the
        page, or None if a cell is missing.

    Raises:
        ValueError: Table, tfoot, or totals row not found.
    """
    table_id = f"box-{team_abbr}-game-basic"
    table = soup.find("table", id=table_id)
    if table is None:
        raise ValueError(f"Box score table not found: #{table_id}")

    tfoot = table.find("tfoot")
    if tfoot is None:
        raise ValueError(f"No <tfoot> in #{table_id}")

    row = tfoot.find("tr")
    if row is None:
        raise ValueError(f"No <tr> in <tfoot> of #{table_id}")

    stats: dict[str, str | None] = {}
    for data_stat, col_name in _BBREF_STAT_TO_COL.items():
        cell = row.find("td", {"data-stat": data_stat})
        if cell is None:
            stats[col_name] = None
            log.debug("Missing cell data-stat=%s for %s", data_stat, team_abbr)
        else:
            text = cell.get_text(strip=True)
            stats[col_name] = text if text else None

    return stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, season: int) -> None:
    """Log warnings for anomalies in a scraped season DataFrame.

    Does not raise — scraping is inherently noisy and a few bad rows
    should not kill the whole run.  Anomalies are flagged for manual review.
    """
    null_pts = df["HOME_PTS"].isna().sum() + df["AWAY_PTS"].isna().sum()
    if null_pts:
        log.warning("Season %d: %d null PTS value(s) — check raw HTML", season, null_pts)

    bad_home = ~df["HOME"].isin(BBREF_TEAMS)
    bad_away = ~df["AWAY"].isin(BBREF_TEAMS)
    if bad_home.any():
        log.warning("Season %d: unknown HOME team(s): %s",
                    season, df.loc[bad_home, "HOME"].unique().tolist())
    if bad_away.any():
        log.warning("Season %d: unknown AWAY team(s): %s",
                    season, df.loc[bad_away, "AWAY"].unique().tolist())

    # NBA regular season has 1,230 games (82 games × 30 teams / 2).
    # Fewer than 500 almost certainly means something went wrong.
    if len(df) < 500:
        log.warning(
            "Season %d: only %d game(s) scraped — expected ~1230. "
            "Page structure may have changed.",
            season, len(df),
        )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """Return a requests.Session with a plausible browser User-Agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return session

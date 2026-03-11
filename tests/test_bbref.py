"""Tests for src/data/scrapers/bbref.py.

All tests are pure unit tests — no network calls are made.
HTTP is mocked at the requests.Session level for integration tests;
parser functions are tested directly with fixture HTML strings.

Structure: happy path + most-likely real failure mode per function,
as required by the engineering rules.
"""

import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.scrapers.bbref import (
    EXPECTED_COLUMNS,
    _extract_date_from_url,
    _extract_team_abbrs,
    _parse_box_score,
    _parse_game_urls,
    _parse_team_totals,
    scrape_season,
)
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

def _schedule_html(include_future_game: bool = False, include_thead_row: bool = True) -> str:
    """Minimal schedule page HTML matching Basketball Reference's structure."""
    future_row = ""
    if include_future_game:
        future_row = """
        <tr>
          <td data-stat="date_game">April 15, 2024</td>
          <td data-stat="visitor_team_name">Boston Celtics</td>
          <td data-stat="pts_a"></td>
          <td data-stat="home_team_name">Miami Heat</td>
          <td data-stat="pts"></td>
          <td data-stat="box_score_text"></td>
        </tr>"""

    month_header = ""
    if include_thead_row:
        month_header = '<tr class="thead"><td colspan="10">October 2023</td></tr>'

    return textwrap.dedent(f"""
        <html><body>
        <table id="schedule">
          <thead><tr><th>Date</th></tr></thead>
          <tbody>
            {month_header}
            <tr>
              <td data-stat="date_game">October 24, 2023</td>
              <td data-stat="visitor_team_name">Los Angeles Lakers</td>
              <td data-stat="pts_a">107</td>
              <td data-stat="home_team_name">Denver Nuggets</td>
              <td data-stat="pts">119</td>
              <td data-stat="box_score_text">
                <a href="/boxscores/202310240DEN.html">Box Score</a>
              </td>
            </tr>
            <tr>
              <td data-stat="date_game">October 25, 2023</td>
              <td data-stat="visitor_team_name">Golden State Warriors</td>
              <td data-stat="pts_a">104</td>
              <td data-stat="home_team_name">Phoenix Suns</td>
              <td data-stat="pts">108</td>
              <td data-stat="box_score_text">
                <a href="/boxscores/202310250PHO.html">Box Score</a>
              </td>
            </tr>
            {future_row}
          </tbody>
        </table>
        </body></html>
    """)


def _box_score_html(
    away: str = "LAL",
    home: str = "BOS",
    away_pts: str = "110",
    home_pts: str = "105",
    missing_tfoot: str | None = None,   # team abbr whose tfoot to remove
) -> str:
    """Minimal box score page HTML matching Basketball Reference's structure.

    Args:
        away: Away team abbreviation.
        home: Home team abbreviation.
        away_pts / home_pts: PTS values placed in the tfoot totals row.
        missing_tfoot: If set to a team abbr, omit that team's <tfoot>.
    """
    def _tfoot(abbr: str, pts: str) -> str:
        if missing_tfoot == abbr:
            return ""
        return textwrap.dedent(f"""
            <tfoot>
              <tr>
                <td data-stat="mp">240:00</td>
                <td data-stat="fg">40</td>
                <td data-stat="fga">88</td>
                <td data-stat="fg_pct">.455</td>
                <td data-stat="fg3">12</td>
                <td data-stat="fg3a">33</td>
                <td data-stat="fg3_pct">.364</td>
                <td data-stat="ft">18</td>
                <td data-stat="fta">22</td>
                <td data-stat="ft_pct">.818</td>
                <td data-stat="orb">7</td>
                <td data-stat="drb">34</td>
                <td data-stat="trb">41</td>
                <td data-stat="ast">24</td>
                <td data-stat="stl">6</td>
                <td data-stat="blk">5</td>
                <td data-stat="tov">11</td>
                <td data-stat="pf">17</td>
                <td data-stat="pts">{pts}</td>
              </tr>
            </tfoot>
        """)

    return textwrap.dedent(f"""
        <html><body>
        <div class="scorebox">
          <div>
            <strong><a href="/teams/{away}/2024.html">{away} Full Name</a></strong>
            <div class="score">{away_pts}</div>
          </div>
          <div>
            <strong><a href="/teams/{home}/2024.html">{home} Full Name</a></strong>
            <div class="score">{home_pts}</div>
          </div>
        </div>

        <table id="box-{away}-game-basic">
          <thead><tr><th>Player</th></tr></thead>
          <tbody><tr><td data-stat="player">LeBron James</td></tr></tbody>
          {_tfoot(away, away_pts)}
        </table>

        <table id="box-{home}-game-basic">
          <thead><tr><th>Player</th></tr></thead>
          <tbody><tr><td data-stat="player">Jayson Tatum</td></tr></tbody>
          {_tfoot(home, home_pts)}
        </table>
        </body></html>
    """)


# ---------------------------------------------------------------------------
# _parse_game_urls
# ---------------------------------------------------------------------------

class TestParseGameUrls:
    def test_happy_path_returns_correct_urls(self):
        urls = _parse_game_urls(_schedule_html())
        assert len(urls) == 2
        assert "https://www.basketball-reference.com/boxscores/202310240DEN.html" in urls
        assert "https://www.basketball-reference.com/boxscores/202310250PHO.html" in urls

    def test_skips_month_separator_thead_rows(self):
        # The result should still only contain the 2 real game rows
        urls = _parse_game_urls(_schedule_html(include_thead_row=True))
        assert len(urls) == 2

    def test_skips_future_games_with_no_box_score_link(self):
        urls = _parse_game_urls(_schedule_html(include_future_game=True))
        # Future game row has no anchor → should not be included
        assert len(urls) == 2

    def test_missing_schedule_table_returns_empty_list(self):
        html = "<html><body><p>No table here</p></body></html>"
        urls = _parse_game_urls(html)
        assert urls == []

    def test_empty_tbody_returns_empty_list(self):
        html = '<html><body><table id="schedule"><tbody></tbody></table></body></html>'
        urls = _parse_game_urls(html)
        assert urls == []


# ---------------------------------------------------------------------------
# _extract_date_from_url
# ---------------------------------------------------------------------------

class TestExtractDateFromUrl:
    def test_standard_url_format(self):
        url = "https://www.basketball-reference.com/boxscores/202310240DEN.html"
        assert _extract_date_from_url(url) == "2023-10-24"

    def test_january_url(self):
        url = "https://www.basketball-reference.com/boxscores/202401150LAL.html"
        assert _extract_date_from_url(url) == "2024-01-15"

    def test_malformed_url_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _extract_date_from_url("https://www.basketball-reference.com/boxscores/bad.html")


# ---------------------------------------------------------------------------
# _extract_team_abbrs
# ---------------------------------------------------------------------------

class TestExtractTeamAbbrs:
    def test_happy_path_returns_away_then_home(self):
        soup = BeautifulSoup(_box_score_html("LAL", "BOS"), "html.parser")
        away, home = _extract_team_abbrs(soup, "http://example.com")
        assert away == "LAL"
        assert home == "BOS"

    def test_no_scorebox_raises_value_error(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        with pytest.raises(ValueError, match="scorebox"):
            _extract_team_abbrs(soup, "http://example.com")

    def test_only_one_team_link_raises_value_error(self):
        html = textwrap.dedent("""
            <html><body>
            <div class="scorebox">
              <div><a href="/teams/LAL/2024.html">LAL</a></div>
            </div>
            </body></html>
        """)
        soup = BeautifulSoup(html, "html.parser")
        with pytest.raises(ValueError, match="2 team abbreviations"):
            _extract_team_abbrs(soup, "http://example.com")

    def test_duplicate_links_deduped_correctly(self):
        # Some pages repeat the team link; we should still get exactly 2 unique abbrs.
        html = textwrap.dedent("""
            <html><body>
            <div class="scorebox">
              <div>
                <a href="/teams/LAL/2024.html">LAL</a>
                <a href="/teams/LAL/2024.html">LAL again</a>
              </div>
              <div><a href="/teams/BOS/2024.html">BOS</a></div>
            </div>
            </body></html>
        """)
        soup = BeautifulSoup(html, "html.parser")
        away, home = _extract_team_abbrs(soup, "http://example.com")
        assert away == "LAL"
        assert home == "BOS"


# ---------------------------------------------------------------------------
# _parse_team_totals
# ---------------------------------------------------------------------------

class TestParseTeamTotals:
    def test_happy_path_returns_all_19_stat_columns(self):
        soup = BeautifulSoup(_box_score_html(), "html.parser")
        stats = _parse_team_totals(soup, "LAL")
        expected_cols = [
            "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
            "FT", "FTA", "FT%", "ORB", "DRB", "TRB",
            "AST", "STL", "BLK", "TO", "PF", "PTS",
        ]
        for col in expected_cols:
            assert col in stats

    def test_tov_remapped_to_to(self):
        # bbref uses data-stat="tov"; our schema calls it "TO"
        soup = BeautifulSoup(_box_score_html(), "html.parser")
        stats = _parse_team_totals(soup, "LAL")
        assert "TO" in stats
        assert "TOV" not in stats

    def test_pts_value_correct(self):
        soup = BeautifulSoup(_box_score_html(away_pts="110", home_pts="105"), "html.parser")
        away_stats = _parse_team_totals(soup, "LAL")
        assert away_stats["PTS"] == "110"

    def test_missing_table_raises_value_error(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        with pytest.raises(ValueError, match="not found"):
            _parse_team_totals(soup, "LAL")

    def test_missing_tfoot_raises_value_error(self):
        soup = BeautifulSoup(_box_score_html(missing_tfoot="LAL"), "html.parser")
        with pytest.raises(ValueError, match="tfoot"):
            _parse_team_totals(soup, "LAL")


# ---------------------------------------------------------------------------
# _parse_box_score (integration of above parsers)
# ---------------------------------------------------------------------------

class TestParseBoxScore:
    def test_happy_path_produces_all_expected_columns(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        row = _parse_box_score(_box_score_html("LAL", "BOS"), url, 2024)
        for col in EXPECTED_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_home_away_assigned_correctly(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        row = _parse_box_score(_box_score_html("LAL", "BOS"), url, 2024)
        assert row["AWAY"] == "LAL"
        assert row["HOME"] == "BOS"

    def test_date_extracted_correctly(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        row = _parse_box_score(_box_score_html("LAL", "BOS"), url, 2024)
        assert row["DATE"] == "2024-01-01"

    def test_season_set_correctly(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        row = _parse_box_score(_box_score_html("LAL", "BOS"), url, 2024)
        assert row["SEASON"] == 2024

    def test_pts_values_correct(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        row = _parse_box_score(
            _box_score_html("LAL", "BOS", away_pts="112", home_pts="98"), url, 2024
        )
        assert row["AWAY_PTS"] == "112"
        assert row["HOME_PTS"] == "98"

    def test_no_scorebox_raises_value_error(self):
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        with pytest.raises(ValueError, match="scorebox"):
            _parse_box_score("<html><body></body></html>", url, 2024)

    def test_missing_box_score_table_raises_value_error(self):
        # Page has a scorebox but the LAL basic table is absent
        html = textwrap.dedent("""
            <html><body>
            <div class="scorebox">
              <div><a href="/teams/LAL/2024.html">LAL</a></div>
              <div><a href="/teams/BOS/2024.html">BOS</a></div>
            </div>
            </body></html>
        """)
        url = "https://www.basketball-reference.com/boxscores/202401010BOS.html"
        with pytest.raises(ValueError, match="not found"):
            _parse_box_score(html, url, 2024)


# ---------------------------------------------------------------------------
# scrape_season — integration test with fully mocked HTTP
# ---------------------------------------------------------------------------

class TestScrapeSeason:
    def _mock_session(self, schedule_html: str, box_html: str) -> MagicMock:
        """Build a mock Session whose .get() returns preset HTML per URL."""
        mock_resp_schedule = MagicMock()
        mock_resp_schedule.status_code = 200
        mock_resp_schedule.text = schedule_html
        mock_resp_schedule.raise_for_status = MagicMock()

        mock_resp_box = MagicMock()
        mock_resp_box.status_code = 200
        mock_resp_box.text = box_html
        mock_resp_box.raise_for_status = MagicMock()

        # All /leagues/ URLs → schedule; all /boxscores/ URLs → box score
        def fake_get(url, **kwargs):
            if "/leagues/" in url:
                return mock_resp_schedule
            return mock_resp_box

        session = MagicMock()
        session.get.side_effect = fake_get
        return session

    @patch("src.data.scrapers.bbref.time.sleep")   # suppress rate-limit delay
    def test_returns_dataframe_with_expected_columns(self, _sleep):
        session = self._mock_session(
            _schedule_html(),
            _box_score_html("LAL", "DEN"),
        )
        df = scrape_season(2024, session=session)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == EXPECTED_COLUMNS

    @patch("src.data.scrapers.bbref.time.sleep")
    def test_returns_one_row_per_completed_game(self, _sleep):
        # Schedule has 2 completed games; future game row has no link
        session = self._mock_session(
            _schedule_html(include_future_game=True),
            _box_score_html("LAL", "DEN"),
        )
        df = scrape_season(2024, session=session)
        # 2 completed games × 7 months = 14 rows (one per url hit)
        assert len(df) == 2 * len([m for m in range(7)])   # 14

    @patch("src.data.scrapers.bbref.time.sleep")
    def test_returns_empty_dataframe_when_all_months_404(self, _sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = mock_resp

        df = scrape_season(2099, session=session)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == EXPECTED_COLUMNS

    @patch("src.data.scrapers.bbref.time.sleep")
    def test_network_error_on_box_score_page_skips_game(self, _sleep):
        mock_schedule = MagicMock()
        mock_schedule.status_code = 200
        mock_schedule.text = _schedule_html()
        mock_schedule.raise_for_status = MagicMock()

        import requests as req
        def fake_get(url, **kwargs):
            if "/leagues/" in url:
                return mock_schedule
            raise req.RequestException("timeout")

        session = MagicMock()
        session.get.side_effect = fake_get

        df = scrape_season(2024, session=session)
        # All box score fetches fail → empty result
        assert df.empty

    @patch("src.data.scrapers.bbref.time.sleep")
    def test_parse_error_on_box_score_page_skips_game(self, _sleep):
        # Box score page exists but returns garbage HTML
        session = self._mock_session(
            _schedule_html(),
            "<html><body>not a box score</body></html>",
        )
        df = scrape_season(2024, session=session)
        assert df.empty

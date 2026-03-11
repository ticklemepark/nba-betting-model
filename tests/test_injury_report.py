"""Tests for src/data/scrapers/injury_report.py."""

import pandas as pd
import pytest

from src.data.scrapers.injury_report import parse_injury_report, EXPECTED_COLUMNS


def _make_espn_html(rows: list[dict]) -> str:
    """Build minimal ESPN-style injury page HTML."""
    team_sections = []
    teams = {}
    for r in rows:
        teams.setdefault(r["team"], []).append(r)

    for team, players in teams.items():
        player_rows = "".join(
            f"<tr><td>{p['name']}</td><td>{p['status']}</td><td>{p.get('reason', 'knee')}</td></tr>"
            for p in players
        )
        team_sections.append(f"""
        <h2 class="TeamName">{team}</h2>
        <div class="ResponsiveTable">
          <table><tbody>{player_rows}</tbody></table>
        </div>
        """)

    return f"<html><body>{''.join(team_sections)}</body></html>"


class TestParseInjuryReport:
    def test_happy_path_returns_dataframe(self):
        html = _make_espn_html([
            {"team": "Los Angeles Lakers", "name": "LeBron James", "status": "Out"},
        ])
        result = parse_injury_report(html)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_expected_columns_present(self):
        html = _make_espn_html([
            {"team": "Los Angeles Lakers", "name": "LeBron James", "status": "Out"},
        ])
        result = parse_injury_report(html)
        for col in EXPECTED_COLUMNS:
            assert col in result.columns

    def test_status_normalised_to_uppercase(self):
        html = _make_espn_html([
            {"team": "Los Angeles Lakers", "name": "LeBron James", "status": "out"},
        ])
        result = parse_injury_report(html)
        assert result.iloc[0]["STATUS"] == "OUT"

    def test_questionable_normalised(self):
        html = _make_espn_html([
            {"team": "Boston Celtics", "name": "Jayson Tatum", "status": "questionable"},
        ])
        result = parse_injury_report(html)
        assert result.iloc[0]["STATUS"] == "QUESTIONABLE"

    def test_multiple_teams(self):
        html = _make_espn_html([
            {"team": "Los Angeles Lakers", "name": "LeBron James", "status": "Out"},
            {"team": "Boston Celtics",     "name": "Jayson Tatum", "status": "Out"},
        ])
        result = parse_injury_report(html)
        assert len(result) == 2
        assert set(result["TEAM"]) == {"Los Angeles Lakers", "Boston Celtics"}

    def test_empty_html_returns_empty_dataframe(self):
        result = parse_injury_report("<html><body></body></html>")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        for col in EXPECTED_COLUMNS:
            assert col in result.columns

    def test_player_name_captured(self):
        html = _make_espn_html([
            {"team": "Los Angeles Lakers", "name": "Anthony Davis", "status": "Doubtful"},
        ])
        result = parse_injury_report(html)
        assert result.iloc[0]["PLAYER_NAME"] == "Anthony Davis"

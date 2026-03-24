"""Tests for src/data/scrapers/injury_report.py."""

import pandas as pd
import pytest

from src.data.scrapers.injury_report import (
    parse_injury_report,
    parse_injury_json,
    EXPECTED_COLUMNS,
)


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


def _make_espn_json(teams: list[dict]) -> dict:
    """Build minimal ESPN-style injury JSON response.

    Args:
        teams: list of {"name": "Atlanta Hawks", "players": [{"name": "...", "status": "...", "reason": "..."}, ...]}
    """
    injury_entries = []
    for team in teams:
        injuries = []
        for p in team.get("players", []):
            injuries.append({
                "athlete": {"displayName": p["name"]},
                "status":  p.get("status", "Out"),
                "shortComment": p.get("reason", ""),
            })
        injury_entries.append({
            "displayName": team["name"],
            "injuries":    injuries,
        })
    return {"injuries": injury_entries}


class TestParseInjuryJson:
    def test_happy_path_returns_dataframe(self):
        data = _make_espn_json([
            {"name": "Los Angeles Lakers", "players": [{"name": "LeBron James", "status": "Out"}]},
        ])
        result = parse_injury_json(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_expected_columns_present(self):
        data = _make_espn_json([
            {"name": "Los Angeles Lakers", "players": [{"name": "LeBron James", "status": "Out"}]},
        ])
        result = parse_injury_json(data)
        for col in EXPECTED_COLUMNS:
            assert col in result.columns

    def test_team_mapped_to_abbreviation(self):
        data = _make_espn_json([
            {"name": "Los Angeles Lakers", "players": [{"name": "LeBron James", "status": "Out"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["TEAM"] == "LAL"

    def test_brooklyn_nets_abbr(self):
        """BRK (bbref) should map to BKN (nba_api)."""
        data = _make_espn_json([
            {"name": "Brooklyn Nets", "players": [{"name": "Ben Simmons", "status": "Out"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["TEAM"] == "BKN"

    def test_charlotte_abbr(self):
        """CHO (bbref) should map to CHA (nba_api)."""
        data = _make_espn_json([
            {"name": "Charlotte Hornets", "players": [{"name": "LaMelo Ball", "status": "Questionable"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["TEAM"] == "CHA"

    def test_phoenix_abbr(self):
        """PHO (bbref) should map to PHX (nba_api)."""
        data = _make_espn_json([
            {"name": "Phoenix Suns", "players": [{"name": "Kevin Durant", "status": "Out"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["TEAM"] == "PHX"

    def test_status_normalised(self):
        data = _make_espn_json([
            {"name": "Boston Celtics", "players": [{"name": "Jayson Tatum", "status": "questionable"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["STATUS"] == "QUESTIONABLE"

    def test_day_to_day_normalised(self):
        data = _make_espn_json([
            {"name": "Miami Heat", "players": [{"name": "Jimmy Butler", "status": "day-to-day"}]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["STATUS"] == "QUESTIONABLE"

    def test_multiple_teams_and_players(self):
        data = _make_espn_json([
            {"name": "Los Angeles Lakers", "players": [
                {"name": "LeBron James", "status": "Out"},
                {"name": "Anthony Davis", "status": "Questionable"},
            ]},
            {"name": "Boston Celtics", "players": [
                {"name": "Jayson Tatum", "status": "Probable"},
            ]},
        ])
        result = parse_injury_json(data)
        assert len(result) == 3
        assert set(result["TEAM"]) == {"LAL", "BOS"}

    def test_reason_captured(self):
        data = _make_espn_json([
            {"name": "Denver Nuggets", "players": [
                {"name": "Nikola Jokic", "status": "Out", "reason": "Left knee soreness"},
            ]},
        ])
        result = parse_injury_json(data)
        assert result.iloc[0]["REASON"] == "Left knee soreness"

    def test_empty_injuries_list_returns_empty_dataframe(self):
        result = parse_injury_json({"injuries": []})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        for col in EXPECTED_COLUMNS:
            assert col in result.columns

    def test_missing_injuries_key_returns_empty_dataframe(self):
        result = parse_injury_json({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

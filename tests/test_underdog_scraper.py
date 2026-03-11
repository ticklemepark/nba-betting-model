"""Tests for src/data/scrapers/underdog.py -- all HTTP calls are mocked.

Fixtures use the public /beta/v6/over_under_lines response schema:
  - appearances, games, over_under_lines are all LISTS (not dicts).
  - games have an integer id and a sport_id field for NBA filtering.
  - american_price in options is optional (defaults to 0.5 when absent).
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.data.scrapers.underdog import (
    STAT_MAP,
    UnderdogAuthError,
    UnderdogAPIError,
    UnderdogGameLine,
    UnderdogPropLine,
    _american_to_prob,
    _normalize_stat,
    fetch_game_lines,
    fetch_prop_lines,
)


# ---------------------------------------------------------------------------
# Minimal fixture matching /beta/v6/over_under_lines (public endpoint)
# ---------------------------------------------------------------------------

def _minimal_prop_response():
    """Minimal valid response from the public over_under_lines endpoint."""
    return {
        "appearances": [
            {
                "id":        "app-uuid-1",
                "player_id": "player-uuid-1",
                "match_id":  131268,          # integer, not string
                "team_id":   "team-bos-uuid",
            }
        ],
        "games": [
            {
                "id":           131268,       # integer key
                "sport_id":     "NBA",        # required for NBA filter
                "title":        "DAL @ BOS",  # "title" field (public endpoint)
                "home_team_id": "team-bos-uuid",
                "away_team_id": "team-dal-uuid",
                "scheduled_at": "2026-03-07T00:00:00Z",
            }
        ],
        "over_under_lines": [
            {
                "id":         "line-uuid-1",
                "status":     "active",
                "stat_value": "25.5",
                "over_under": {
                    "appearance_stat": {
                        "appearance_id": "app-uuid-1",
                        "stat":          "points",
                    },
                },
                "options": [
                    {
                        "choice":           "higher",
                        "american_price":   "-112",
                        "selection_header": "Jayson Tatum",
                        "status":           "active",
                    },
                    {
                        "choice":           "lower",
                        "american_price":   "-112",
                        "selection_header": "Jayson Tatum",
                        "status":           "active",
                    },
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# _american_to_prob
# ---------------------------------------------------------------------------

class TestAmericanToProb:
    def test_negative_odds(self):
        # -110 -> 110/210 = 0.5238
        p = _american_to_prob("-110")
        assert abs(p - 110 / 210) < 0.001

    def test_positive_odds(self):
        # +100 -> 100/200 = 0.5
        p = _american_to_prob("+100")
        assert abs(p - 0.5) < 0.001

    def test_minus_125(self):
        p = _american_to_prob("-125")
        assert abs(p - 125 / 225) < 0.001

    def test_plus_102(self):
        p = _american_to_prob("+102")
        assert abs(p - 100 / 202) < 0.001

    def test_output_in_zero_one(self):
        for price in ["-500", "-200", "-110", "+100", "+200", "+500"]:
            assert 0.0 < _american_to_prob(price) < 1.0


# ---------------------------------------------------------------------------
# _normalize_stat
# ---------------------------------------------------------------------------

class TestNormalizeStat:
    def test_real_stat_names_map_correctly(self):
        assert _normalize_stat("points")              == "PTS"
        assert _normalize_stat("rebounds")            == "REB"
        assert _normalize_stat("assists")             == "AST"
        assert _normalize_stat("pts_rebs_asts")       == "PRA"
        assert _normalize_stat("three_pointers_made") == "FG3M"
        assert _normalize_stat("steals")              == "STL"
        assert _normalize_stat("blocks")              == "BLK"
        assert _normalize_stat("turnovers")           == "TOV"

    def test_legacy_aliases_still_map(self):
        assert _normalize_stat("pts_reb_ast") == "PRA"
        assert _normalize_stat("reb_ast")     == "RA"
        assert _normalize_stat("three_pt_fg") == "FG3M"

    def test_unknown_stat_uppercased(self):
        assert _normalize_stat("fantasy_score") == "FANTASY_SCORE"

    def test_case_insensitive(self):
        assert _normalize_stat("POINTS")   == "PTS"
        assert _normalize_stat("Rebounds") == "REB"


# ---------------------------------------------------------------------------
# fetch_prop_lines
# ---------------------------------------------------------------------------

class TestFetchPropLines:
    def _mock_get(self, payload):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        return mock_resp

    @patch("src.data.scrapers.underdog.requests.get")
    def test_returns_list_of_prop_lines(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines("2026-03-07")
        assert len(lines) == 1
        assert isinstance(lines[0], UnderdogPropLine)

    @patch("src.data.scrapers.underdog.requests.get")
    def test_stat_normalized_to_pts(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        assert lines[0].stat == "PTS"

    @patch("src.data.scrapers.underdog.requests.get")
    def test_line_value_correct(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        assert lines[0].line == 25.5

    @patch("src.data.scrapers.underdog.requests.get")
    def test_payout_from_american_price(self, mock_get):
        # Both sides -112 -> 112/212 ~= 0.5283
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        expected = 112.0 / 212.0
        assert abs(lines[0].over_payout  - expected) < 0.001
        assert abs(lines[0].under_payout - expected) < 0.001

    @patch("src.data.scrapers.underdog.requests.get")
    def test_player_name_from_selection_header(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        assert lines[0].player_name == "Jayson Tatum"

    @patch("src.data.scrapers.underdog.requests.get")
    def test_team_and_opp_resolved_from_game(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        # BOS is home (team_id matches home_team_id), DAL is away
        assert lines[0].team == "BOS"
        assert lines[0].opp  == "DAL"

    @patch("src.data.scrapers.underdog.requests.get")
    def test_game_id_is_match_id(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        assert lines[0].game_id == "131268"

    @patch("src.data.scrapers.underdog.requests.get")
    def test_game_date_parsed_from_scheduled_at(self, mock_get):
        mock_get.return_value = self._mock_get(_minimal_prop_response())
        lines = fetch_prop_lines()
        assert lines[0].game_date == date(2026, 3, 7)

    @patch("src.data.scrapers.underdog.requests.get")
    def test_inactive_lines_skipped(self, mock_get):
        payload = _minimal_prop_response()
        payload["over_under_lines"][0]["status"] = "inactive"
        mock_get.return_value = self._mock_get(payload)
        lines = fetch_prop_lines()
        assert lines == []

    @patch("src.data.scrapers.underdog.requests.get")
    def test_non_nba_games_skipped(self, mock_get):
        payload = _minimal_prop_response()
        payload["games"][0]["sport_id"] = "NFL"
        mock_get.return_value = self._mock_get(payload)
        lines = fetch_prop_lines()
        assert lines == []

    @patch("src.data.scrapers.underdog.requests.get")
    def test_missing_american_price_defaults_to_half(self, mock_get):
        payload = _minimal_prop_response()
        for opt in payload["over_under_lines"][0]["options"]:
            opt.pop("american_price", None)
        mock_get.return_value = self._mock_get(payload)
        lines = fetch_prop_lines()
        assert lines[0].over_payout  == 0.5
        assert lines[0].under_payout == 0.5

    @patch("src.data.scrapers.underdog.requests.get")
    def test_401_raises_auth_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp
        with pytest.raises(UnderdogAuthError):
            fetch_prop_lines()

    @patch("src.data.scrapers.underdog.requests.get")
    def test_500_raises_api_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        mock_get.return_value = mock_resp
        with pytest.raises(UnderdogAPIError):
            fetch_prop_lines()

    @patch("src.data.scrapers.underdog.requests.get")
    def test_empty_response_returns_empty_list(self, mock_get):
        mock_get.return_value = self._mock_get(
            {"appearances": [], "games": [], "over_under_lines": []}
        )
        assert fetch_prop_lines() == []

    @patch("src.data.scrapers.underdog.requests.get")
    def test_line_missing_options_skipped(self, mock_get):
        payload = _minimal_prop_response()
        # Add a second line with no options — should be skipped.
        payload["over_under_lines"].append({
            "id": "bad-line", "status": "active", "stat_value": "8.0",
            "over_under": {"appearance_stat": {"appearance_id": "app-uuid-1",
                                               "stat": "rebounds"}},
            "options": [],
        })
        mock_get.return_value = self._mock_get(payload)
        lines = fetch_prop_lines()
        assert all(l.stat == "PTS" for l in lines)


# ---------------------------------------------------------------------------
# fetch_game_lines  (Rival endpoint not yet implemented -> returns [])
# ---------------------------------------------------------------------------

class TestFetchGameLines:
    def test_returns_empty_list(self):
        lines = fetch_game_lines()
        assert lines == []

    def test_returns_empty_list_with_date(self):
        lines = fetch_game_lines("2026-03-07")
        assert lines == []

    def test_returns_list_type(self):
        assert isinstance(fetch_game_lines(), list)

"""NBA team constants and abbreviation mappings.

All abbreviations use Basketball Reference's conventions, which differ from
nba_api's in three cases:
    BRK (bbref) = BKN (nba_api)  — Brooklyn Nets
    CHO (bbref) = CHA (nba_api)  — Charlotte Hornets
    PHO (bbref) = PHX (nba_api)  — Phoenix Suns
"""

# ---------------------------------------------------------------------------
# Basketball Reference team abbreviations (2015–2024 seasons)
# ---------------------------------------------------------------------------

BBREF_TEAMS: frozenset[str] = frozenset({
    "ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHO",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})

# ---------------------------------------------------------------------------
# Cross-reference maps for nba_api integration (Phase 2)
# ---------------------------------------------------------------------------

# bbref abbreviation → nba_api abbreviation (only entries that differ)
BBREF_TO_NBA_API: dict[str, str] = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",
}

# nba_api abbreviation → bbref abbreviation (inverse of above)
NBA_API_TO_BBREF: dict[str, str] = {v: k for k, v in BBREF_TO_NBA_API.items()}

# ---------------------------------------------------------------------------
# Full team names keyed by Basketball Reference abbreviation
# ---------------------------------------------------------------------------

TEAM_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls",
    "CHO": "Charlotte Hornets",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

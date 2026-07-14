"""Evidence packets: the ONLY facts a narrative is allowed to cite.

An EvidencePacket is a flat dict of verified values pulled straight from
the pick objects, the player feature parquet, and today's injury report.
The LLM prompt is built from this packet, and the verifier checks the
generated text against it.  Nothing outside the packet is citable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Human-readable stat names used in prompts and templates.
STAT_NAMES: dict[str, str] = {
    "PTS":  "points",
    "REB":  "rebounds",
    "AST":  "assists",
    "FG3M": "three-pointers made",
    "STL":  "steals",
    "BLK":  "blocks",
    "TOV":  "turnovers",
    "PRA":  "points+rebounds+assists",
    "PR":   "points+rebounds",
    "PA":   "points+assists",
    "RA":   "rebounds+assists",
}

# Rolling-window sizes are citable ("last 5 games") even though they are
# constants rather than facts.
CITABLE_CONSTANTS = (2, 3, 5, 10, 20)


@dataclass
class EvidencePacket:
    """Structured, verified facts for one pick."""

    pick_type: str                       # "prop" or "game"
    facts: dict[str, Any] = field(default_factory=dict)

    def numeric_facts(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.facts.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    def text_facts(self) -> dict[str, str]:
        return {k: v for k, v in self.facts.items() if isinstance(v, str)}


def _clean(value: Any, ndigits: int = 1) -> float | None:
    """Round a numeric value; return None for NaN/None/non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return round(f, ndigits)


def make_prop_evidence(**facts: Any) -> EvidencePacket:
    """Build a prop evidence packet from keyword facts, dropping None/NaN."""
    cleaned: dict[str, Any] = {}
    for k, v in facts.items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        cleaned[k] = v
    return EvidencePacket(pick_type="prop", facts=cleaned)


def build_prop_evidence(
    pick,                                   # PropPick
    player_row: pd.Series | None = None,    # most-recent feature row for this player
    injuries: pd.DataFrame | None = None,   # today's injury report
) -> EvidencePacket:
    """Assemble the evidence packet for a player prop pick.

    Every value comes from the pick object or the feature row — the same
    inputs the model itself used.  Direction-side probability is expressed
    as a percentage so narratives can say "58.2% chance".
    """
    stat = pick.stat
    prob_over = float(pick.model_prob_over)
    prob_pick_side = prob_over if pick.direction == "over" else 1.0 - prob_over
    ud_over = float(pick.underdog_prob_over)
    ud_pick_side = ud_over if pick.direction == "over" else 1.0 - ud_over

    facts: dict[str, Any] = dict(
        player_name       = pick.player_name,
        team              = pick.team,
        opponent          = pick.opp,
        stat              = stat,
        stat_name         = STAT_NAMES.get(stat, stat.lower()),
        direction         = pick.direction,
        line              = _clean(pick.line),
        model_median      = _clean(pick.model_median),
        model_low         = _clean(pick.model_low),
        model_high        = _clean(pick.model_high),
        model_prob_pct    = _clean(prob_pick_side * 100),
        underdog_prob_pct = _clean(ud_pick_side * 100),
        edge_pp           = _clean(abs(pick.edge) * 100),
        game_date         = str(pick.game_date),
    )

    if player_row is not None:
        row = player_row
        col_map = {
            f"{stat}_L5":         "avg_l5",
            f"{stat}_L10":        "avg_l10",
            f"{stat}_SEASON":     "avg_season",
            f"{stat}_HOME_AVG":   "home_avg",
            f"{stat}_AWAY_AVG":   "away_avg",
            f"{stat}_VS_OPP_AVG": "vs_opp_avg",
            f"{stat}_VS_OPP_N":   "vs_opp_games",
            "MIN_L5":             "minutes_l5",
            "USAGE_PROXY_L5":     "usage_l5",
            f"{stat}_INJ_DELTA":      "injury_delta",
            f"{stat}_OPP_INJ_DELTA":  "opp_injury_delta",
        }
        for src, dst in col_map.items():
            if src in row.index:
                val = _clean(row[src])
                if val is not None:
                    facts[dst] = val

        if "INJURY_ADJUSTED" in row.index and bool(row["INJURY_ADJUSTED"]):
            facts["injury_adjusted"] = True
            method = row.get("INJURY_METHOD", "")
            if method:
                facts["injury_method"] = str(method)

    # Names of OUT/DOUBTFUL players on this player's team and the opponent.
    if injuries is not None and not injuries.empty and "STATUS" in injuries.columns:
        absent = injuries[injuries["STATUS"].isin(["OUT", "DOUBTFUL"])]
        team_out = absent[absent["TEAM"] == pick.team]["PLAYER_NAME"].tolist()
        opp_out  = absent[absent["TEAM"] == pick.opp]["PLAYER_NAME"].tolist()
        if team_out:
            facts["teammates_out"] = ", ".join(team_out)
        if opp_out:
            facts["opponents_out"] = ", ".join(opp_out)

    return make_prop_evidence(**facts)


def build_game_evidence(
    pick,                                 # GamePick
    game_row: pd.Series | None = None,    # today's game feature row (optional)
) -> EvidencePacket:
    """Assemble the evidence packet for a game outcome pick."""
    prob_home = float(pick.model_prob_home)
    prob_pick_side = prob_home if pick.direction == "home" else 1.0 - prob_home
    ud_home = float(pick.underdog_prob_home)
    ud_pick_side = ud_home if pick.direction == "home" else 1.0 - ud_home
    pick_team = pick.home_team if pick.direction == "home" else pick.away_team

    facts: dict[str, Any] = dict(
        home_team         = pick.home_team,
        away_team         = pick.away_team,
        pick_team         = pick_team,
        direction         = pick.direction,
        model_prob_pct    = _clean(prob_pick_side * 100),
        underdog_prob_pct = _clean(ud_pick_side * 100),
        edge_pp           = _clean(abs(pick.edge) * 100),
        game_date         = str(pick.game_date),
    )

    if game_row is not None:
        col_map = {
            "HOME_AFTER":  "home_elo",
            "AWAY_AFTER":  "away_elo",
            "HOME_ELO":    "home_elo",
            "AWAY_ELO":    "away_elo",
            "HOME_STREAK": "home_streak",
            "AWAY_STREAK": "away_streak",
            "HOME_B2B":    "home_b2b",
            "AWAY_B2B":    "away_b2b",
            "HOME_REC":    "home_h2h_record",
            "AWAY_REC":    "away_h2h_record",
        }
        for src, dst in col_map.items():
            if src in game_row.index and dst not in facts:
                val = _clean(game_row[src])
                if val is not None:
                    facts[dst] = val

    packet = make_prop_evidence(**facts)
    packet.pick_type = "game"
    return packet

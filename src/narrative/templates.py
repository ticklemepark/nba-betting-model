"""Deterministic fallback narratives assembled directly from evidence.

Used when the LLM is unavailable (no API key) or when a generated
narrative fails verification twice.  Because every clause is built from
the evidence packet itself, template output always passes the verifier —
tests enforce this round-trip property.

Style notes (verifier-safe by construction):
- Never uses the word "over"/"under" except for the pick's own direction
  (so we write "in his last 5 games", not "over his last 5 games").
- Only cites numbers present in the evidence packet.
"""

from src.narrative.evidence import EvidencePacket


def _fmt(v: float) -> str:
    """Format a number the same way it is stored in evidence (1 decimal)."""
    return f"{v:g}" if float(v) == int(v) else f"{v:.1f}"


def template_narrative(evidence: EvidencePacket) -> str:
    if evidence.pick_type == "game":
        return _game_template(evidence)
    return _prop_template(evidence)


def _prop_template(e: EvidencePacket) -> str:
    f = e.facts
    direction = str(f.get("direction", "")).upper()
    parts: list[str] = []

    lead = (
        f"{f.get('player_name', 'Player')} {direction} {_fmt(f['line'])} "
        f"{f.get('stat_name', '')} vs {f.get('opponent', '?')}:"
    )
    core = (
        f" the model projects {_fmt(f['model_median'])} "
        f"(range {_fmt(f['model_low'])}–{_fmt(f['model_high'])}), "
        f"a {_fmt(f['model_prob_pct'])}% chance on the {direction} side against "
        f"Underdog's implied {_fmt(f['underdog_prob_pct'])}% — "
        f"a {_fmt(f['edge_pp'])}pp edge."
    )
    parts.append(lead + core)

    form_bits: list[str] = []
    if "avg_l5" in f:
        form_bits.append(f"{_fmt(f['avg_l5'])} in his last 5")
    if "avg_l10" in f:
        form_bits.append(f"{_fmt(f['avg_l10'])} in his last 10")
    if "avg_season" in f:
        form_bits.append(f"{_fmt(f['avg_season'])} on the season")
    if form_bits:
        parts.append(" He is averaging " + ", ".join(form_bits) + ".")

    if "vs_opp_avg" in f and "vs_opp_games" in f:
        parts.append(
            f" Against {f.get('opponent', 'this opponent')} he has averaged "
            f"{_fmt(f['vs_opp_avg'])} across {_fmt(f['vs_opp_games'])} matchups."
        )

    if f.get("injury_adjusted"):
        who = f.get("teammates_out") or f.get("opponents_out")
        if who:
            parts.append(
                f" Projection includes an injury adjustment with {who} listed OUT/DOUBTFUL."
            )
        else:
            parts.append(" Projection includes an injury-based adjustment.")

    return "".join(parts)


def _game_template(e: EvidencePacket) -> str:
    f = e.facts
    parts: list[str] = []
    parts.append(
        f"{f.get('pick_team', '?')} to win "
        f"({f.get('away_team', '?')} at {f.get('home_team', '?')}): "
        f"the model gives a {_fmt(f['model_prob_pct'])}% win probability against "
        f"Underdog's implied {_fmt(f['underdog_prob_pct'])}% — "
        f"a {_fmt(f['edge_pp'])}pp edge."
    )

    if "home_elo" in f and "away_elo" in f:
        parts.append(
            f" ELO ratings: {f.get('home_team', 'home')} {_fmt(f['home_elo'])}, "
            f"{f.get('away_team', 'away')} {_fmt(f['away_elo'])}."
        )
    b2b_bits = []
    if f.get("home_b2b"):
        b2b_bits.append(f"{f.get('home_team', 'home')} is on a back-to-back")
    if f.get("away_b2b"):
        b2b_bits.append(f"{f.get('away_team', 'away')} is on a back-to-back")
    if b2b_bits:
        parts.append(" " + " and ".join(b2b_bits) + ".")

    return "".join(parts)

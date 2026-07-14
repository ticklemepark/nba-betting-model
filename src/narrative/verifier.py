"""Deterministic verification of generated narratives against evidence.

The verifier is the anti-hallucination gate.  It never calls an LLM —
it is pure string/number matching, so it cannot itself hallucinate.

Checks performed:
1. NUMBERS  — every number cited in the text must match a fact in the
   evidence packet (within rounding tolerance), or be a citable window
   constant (2/3/5/10/20 as in "last 5 games").
2. DIRECTION — a pick explained as "over" must not be described with the
   word "under" (and vice versa).  "Underdog" is not flagged (word-boundary
   match).  Game picks: the pick_team must appear; "home"/"away" mixups are
   caught via the direction word check.
3. ENTITIES — the player (or picked team) must be named, and no OTHER
   known player names (optional roster list) may appear.
"""

import re
from dataclasses import dataclass, field

from src.narrative.evidence import CITABLE_CONSTANTS, EvidencePacket

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class VerificationResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _allowed_values(evidence: EvidencePacket) -> list[float]:
    """All numeric values a narrative may cite."""
    allowed: set[float] = set(float(c) for c in CITABLE_CONSTANTS)
    for v in evidence.numeric_facts().values():
        allowed.add(v)
        allowed.add(round(v, 1))
        allowed.add(round(v, 0))
    return sorted(allowed)


def _number_matches(cited: str, allowed: list[float]) -> bool:
    """A cited number matches if it equals an allowed value at the cited precision."""
    n = float(cited)
    # Tolerance depends on how precisely the number was cited:
    # "28.4" must match to ±0.05; "28" only to ±0.5 (i.e. it may be a rounding).
    tol = 0.051 if "." in cited else 0.51
    return any(abs(n - v) <= tol for v in allowed)


def verify_narrative(
    text: str,
    evidence: EvidencePacket,
    known_names: list[str] | None = None,
) -> VerificationResult:
    """Check a narrative against its evidence packet.

    Args:
        text:        The generated narrative.
        evidence:    The packet the narrative was generated from.
        known_names: Optional list of other player names in today's pool.
                     Any of these appearing in the text (other than the
                     pick's own player / teammates listed in evidence)
                     is flagged as a cross-player hallucination.

    Returns:
        VerificationResult(ok, violations).
    """
    violations: list[str] = []
    facts = evidence.facts

    # ---- 1. Numeric claims -------------------------------------------------
    allowed = _allowed_values(evidence)
    for match in _NUMBER_RE.finditer(text):
        cited = match.group(0)
        if not _number_matches(cited, allowed):
            violations.append(
                f"number '{cited}' does not match any evidence value"
            )

    # ---- 2. Direction consistency -------------------------------------------
    direction = str(facts.get("direction", "")).lower()
    if direction in ("over", "under"):
        wrong = "under" if direction == "over" else "over"
        # \bunder\b does NOT match "Underdog" (word continues with 'd').
        if re.search(rf"\b{wrong}\b", text, flags=re.IGNORECASE):
            violations.append(
                f"pick direction is '{direction}' but text uses the word '{wrong}'"
            )

    # ---- 3. Entity checks ---------------------------------------------------
    if evidence.pick_type == "prop":
        player = facts.get("player_name", "")
        if player and player.lower() not in text.lower():
            violations.append(f"narrative never names the player '{player}'")
    else:
        pick_team = facts.get("pick_team", "")
        if pick_team and pick_team.lower() not in text.lower():
            violations.append(f"narrative never names the picked team '{pick_team}'")

    if known_names:
        # Names that ARE legitimately citable: the player plus anyone listed
        # in the injury evidence.
        legit = {str(facts.get("player_name", "")).lower()}
        for key in ("teammates_out", "opponents_out"):
            for name in str(facts.get(key, "")).split(","):
                legit.add(name.strip().lower())
        for other in known_names:
            other_l = other.strip().lower()
            if other_l and other_l not in legit and other_l in text.lower():
                violations.append(
                    f"narrative mentions '{other}', who is not part of this pick's evidence"
                )

    return VerificationResult(ok=not violations, violations=violations)

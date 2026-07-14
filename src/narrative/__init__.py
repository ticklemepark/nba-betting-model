"""LLM narrative layer: evidence-grounded explanations for picks.

Every narrative is generated from a structured EvidencePacket containing
only facts the model pipeline actually produced, then run through a
deterministic verifier that rejects any claim (number, name, direction)
not present in the evidence.  Failed generations fall back to a template
narrative assembled directly from the evidence — never hallucinated.
"""

from src.narrative.evidence import (
    EvidencePacket,
    build_game_evidence,
    build_prop_evidence,
    make_prop_evidence,
)
from src.narrative.generator import NarrativeGenerator, NarrativeResult, narratives_for_picks
from src.narrative.templates import template_narrative
from src.narrative.verifier import VerificationResult, verify_narrative

__all__ = [
    "EvidencePacket",
    "build_prop_evidence",
    "build_game_evidence",
    "make_prop_evidence",
    "NarrativeGenerator",
    "NarrativeResult",
    "narratives_for_picks",
    "template_narrative",
    "VerificationResult",
    "verify_narrative",
]

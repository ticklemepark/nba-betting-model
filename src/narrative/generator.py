"""LLM narrative generation with a verify → retry → fallback loop.

Flow per pick:
    1. Build prompt from the EvidencePacket (the only citable facts).
    2. Call the Anthropic API (model from NARRATIVE_MODEL, default Haiku).
    3. Run the deterministic verifier.
    4. On violation: retry ONCE with the violations fed back.
    5. Still failing (or no API key / SDK / network): fall back to the
       template narrative, which is assembled from evidence and cannot
       hallucinate.

The result records which path produced the text ("llm", "llm-retry",
"template") plus any violations from the final failed LLM attempt, so
you can audit how often the model needed correcting.
"""

import json
import logging
import os
from dataclasses import dataclass, field

from src.narrative.evidence import EvidencePacket
from src.narrative.templates import template_narrative
from src.narrative.verifier import verify_narrative

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 300

_SYSTEM_PROMPT = """You write short, sharp explanations of NBA betting picks for a daily pick sheet.

HARD RULES — violating any of these gets your output rejected:
1. Use ONLY the facts in the EVIDENCE JSON. Do not add any outside knowledge
   about players, teams, injuries, streaks, trades, or news.
2. Every number you write must appear in the evidence (you may round to 1
   decimal place). Never compute new numbers.
3. Never use the word "{forbidden}" — the pick direction is "{direction}".
4. Name the player/team exactly as given. Mention no other players except
   any listed in teammates_out / opponents_out.
5. 2–3 sentences, plain text, no bullet points, no hedging boilerplate.

Write like a sharp analyst explaining WHY the model likes this pick: lead
with the model projection vs. the line, then the strongest supporting facts
(recent form, matchup history, injury context)."""


@dataclass
class NarrativeResult:
    text: str
    source: str                       # "llm" | "llm-retry" | "template"
    verified: bool = True
    violations: list[str] = field(default_factory=list)   # from last failed LLM attempt


class NarrativeGenerator:
    """Generates verified narratives for picks via the Anthropic API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model or os.environ.get("NARRATIVE_MODEL", _DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self._client = None

    # -- availability --------------------------------------------------------

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # -- LLM call (isolated for testability) ----------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()

    # -- main entry point ------------------------------------------------------

    def generate(
        self,
        evidence: EvidencePacket,
        known_names: list[str] | None = None,
    ) -> NarrativeResult:
        """Generate a verified narrative for one pick."""
        if not self.available():
            return NarrativeResult(
                text=template_narrative(evidence),
                source="template",
            )

        direction = str(evidence.facts.get("direction", ""))
        forbidden = {"over": "under", "under": "over"}.get(direction, "")
        system = _SYSTEM_PROMPT.format(direction=direction or "n/a",
                                       forbidden=forbidden or "n/a")
        user = "EVIDENCE JSON:\n" + json.dumps(evidence.facts, indent=2, default=str)

        last_violations: list[str] = []
        for attempt, source in ((1, "llm"), (2, "llm-retry")):
            try:
                text = self._call_llm(system, user)
            except Exception as exc:
                log.warning("Narrative LLM call failed (%s) — using template.", exc)
                return NarrativeResult(text=template_narrative(evidence), source="template")

            check = verify_narrative(text, evidence, known_names=known_names)
            if check.ok:
                return NarrativeResult(text=text, source=source)

            last_violations = check.violations
            log.info("Narrative attempt %d failed verification: %s", attempt, check.violations)
            user += (
                "\n\nYour previous answer was REJECTED by the fact-checker:\n- "
                + "\n- ".join(check.violations)
                + "\nRewrite it using only facts from the EVIDENCE JSON."
            )

        # Two strikes → deterministic fallback.
        return NarrativeResult(
            text=template_narrative(evidence),
            source="template",
            verified=True,
            violations=last_violations,
        )


# ---------------------------------------------------------------------------
# Batch helper for the daily pipeline
# ---------------------------------------------------------------------------

def narratives_for_picks(
    picks: list,
    player_df=None,          # pd.DataFrame of today's player feature rows
    injuries=None,           # pd.DataFrame injury report
    game_df=None,            # pd.DataFrame of today's game feature rows
    generator: NarrativeGenerator | None = None,
) -> dict[str, NarrativeResult]:
    """Generate narratives for a list of unique picks.

    Returns a dict keyed by pick key (see pick_key) so callers can attach
    narratives to entries without regenerating for duplicated picks.
    """
    from src.betting.edge_calculator import GamePick, PropPick
    from src.narrative.evidence import build_game_evidence, build_prop_evidence

    gen = generator or NarrativeGenerator()
    known_names: list[str] = []
    if player_df is not None and not player_df.empty and "PLAYER_NAME" in player_df.columns:
        known_names = player_df["PLAYER_NAME"].dropna().unique().tolist()

    results: dict[str, NarrativeResult] = {}
    for pick in picks:
        key = pick_key(pick)
        if key in results:
            continue

        if isinstance(pick, PropPick):
            row = None
            if player_df is not None and not player_df.empty and "PLAYER_NAME" in player_df.columns:
                matches = player_df[player_df["PLAYER_NAME"] == pick.player_name]
                if not matches.empty:
                    row = matches.iloc[-1]
            evidence = build_prop_evidence(pick, player_row=row, injuries=injuries)
        elif isinstance(pick, GamePick):
            row = None
            if game_df is not None and not game_df.empty and \
                    "HOME" in game_df.columns and "AWAY" in game_df.columns:
                matches = game_df[
                    (game_df["HOME"].astype(str).str.upper() == pick.home_team.upper())
                    & (game_df["AWAY"].astype(str).str.upper() == pick.away_team.upper())
                ]
                if not matches.empty:
                    row = matches.iloc[-1]
            evidence = build_game_evidence(pick, game_row=row)
        else:
            continue

        results[key] = gen.generate(evidence, known_names=known_names)

    return results


def pick_key(pick) -> str:
    """Stable identity key for a pick (used to attach narratives to entries)."""
    from src.betting.edge_calculator import GamePick, PropPick
    if isinstance(pick, PropPick):
        return f"{pick.player_name}|{pick.stat}|{pick.line:g}|{pick.direction}"
    if isinstance(pick, GamePick):
        return f"{pick.game_id}|{pick.direction}"
    return str(pick)

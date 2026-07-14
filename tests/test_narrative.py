"""Tests for the LLM narrative layer (evidence → generate → verify → fallback)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.betting.edge_calculator import GamePick, PropPick
from src.narrative.evidence import build_game_evidence, build_prop_evidence
from src.narrative.generator import NarrativeGenerator, narratives_for_picks, pick_key
from src.narrative.templates import template_narrative
from src.narrative.verifier import verify_narrative


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prop_pick() -> PropPick:
    return PropPick(
        player_name="LeBron James",
        team="LAL",
        opp="BOS",
        game_id="G1",
        stat="PTS",
        direction="over",
        line=25.5,
        model_median=28.4,
        model_low=22.1,
        model_high=33.9,
        model_prob_over=0.612,
        underdog_prob_over=0.50,
        edge=0.112,
        game_date=date(2026, 1, 15),
    )


@pytest.fixture
def under_pick(prop_pick) -> PropPick:
    return PropPick(
        **{**prop_pick.__dict__, "direction": "under",
           "model_prob_over": 0.41, "edge": -0.09},
    )


@pytest.fixture
def player_row() -> pd.Series:
    return pd.Series({
        "PLAYER_NAME": "LeBron James",
        "PTS_L5": 27.9,
        "PTS_L10": 26.8,
        "PTS_SEASON": 26.1,
        "PTS_VS_OPP_AVG": 29.1,
        "PTS_VS_OPP_N": 3,
        "MIN_L5": 35.2,
        "USAGE_PROXY_L5": 31.4,
        "PTS_HOME_AVG": np.nan,          # NaN must be dropped
        "INJURY_ADJUSTED": True,
        "INJURY_METHOD": "historical",
    })


@pytest.fixture
def injuries() -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_NAME": ["Anthony Davis", "Jayson Tatum"],
        "TEAM":        ["LAL", "BOS"],
        "STATUS":      ["OUT", "DOUBTFUL"],
        "REASON":      ["ankle", "knee"],
    })


@pytest.fixture
def game_pick() -> GamePick:
    return GamePick(
        game_id="G2",
        home_team="SAC",
        away_team="LAL",
        direction="home",
        model_prob_home=0.63,
        underdog_prob_home=0.52,
        edge=0.11,
        game_date=date(2026, 1, 15),
    )


# ---------------------------------------------------------------------------
# Evidence packets
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_basic_prop_facts(self, prop_pick, player_row, injuries):
        e = build_prop_evidence(prop_pick, player_row, injuries)
        f = e.facts
        assert f["player_name"] == "LeBron James"
        assert f["line"] == 25.5
        assert f["model_median"] == 28.4
        assert f["model_prob_pct"] == 61.2
        assert f["underdog_prob_pct"] == 50.0
        assert f["edge_pp"] == 11.2
        assert f["avg_l5"] == 27.9
        assert f["injury_adjusted"] is True
        assert "Anthony Davis" in f["teammates_out"]
        assert "Jayson Tatum" in f["opponents_out"]

    def test_nan_facts_dropped(self, prop_pick, player_row):
        e = build_prop_evidence(prop_pick, player_row)
        assert "home_avg" not in e.facts   # was NaN

    def test_under_pick_flips_probability_side(self, under_pick):
        e = build_prop_evidence(under_pick)
        assert e.facts["model_prob_pct"] == pytest.approx(59.0)
        assert e.facts["underdog_prob_pct"] == pytest.approx(50.0)
        assert e.facts["edge_pp"] == pytest.approx(9.0)

    def test_game_evidence(self, game_pick):
        e = build_game_evidence(game_pick)
        assert e.pick_type == "game"
        assert e.facts["pick_team"] == "SAC"
        assert e.facts["model_prob_pct"] == 63.0


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class TestVerifier:
    def test_accepts_grounded_text(self, prop_pick, player_row):
        e = build_prop_evidence(prop_pick, player_row)
        text = (
            "LeBron James is a strong OVER at 25.5 points: the model projects "
            "28.4 with a 61.2% chance to clear, an 11.2pp edge on Underdog's "
            "implied 50%. He is averaging 27.9 in his last 5 games."
        )
        result = verify_narrative(text, e)
        assert result.ok, result.violations

    def test_flags_fabricated_number(self, prop_pick):
        e = build_prop_evidence(prop_pick)
        text = "LeBron James dropped 45 last night and the OVER 25.5 is live."
        result = verify_narrative(text, e)
        assert not result.ok
        assert any("45" in v for v in result.violations)

    def test_flags_wrong_direction_word(self, prop_pick):
        e = build_prop_evidence(prop_pick)
        text = "LeBron James should stay under 25.5 tonight."
        result = verify_narrative(text, e)
        assert not result.ok
        assert any("direction" in v for v in result.violations)

    def test_underdog_word_not_flagged_as_under(self, prop_pick):
        e = build_prop_evidence(prop_pick)
        text = "LeBron James OVER 25.5 — Underdog's line looks soft at 50%."
        result = verify_narrative(text, e)
        assert result.ok, result.violations

    def test_flags_missing_player_name(self, prop_pick):
        e = build_prop_evidence(prop_pick)
        text = "The King goes OVER 25.5 tonight."
        result = verify_narrative(text, e)
        assert not result.ok

    def test_flags_cross_player_mention(self, prop_pick, injuries):
        e = build_prop_evidence(prop_pick, injuries=injuries)
        text = "LeBron James OVER 25.5 because Stephen Curry is resting."
        result = verify_narrative(text, e, known_names=["Stephen Curry", "LeBron James"])
        assert not result.ok
        assert any("Stephen Curry" in v for v in result.violations)

    def test_injured_teammate_mention_allowed(self, prop_pick, injuries):
        e = build_prop_evidence(prop_pick, injuries=injuries)
        text = "With Anthony Davis OUT, LeBron James OVER 25.5 looks strong at 61.2%."
        result = verify_narrative(
            text, e, known_names=["Anthony Davis", "LeBron James"]
        )
        assert result.ok, result.violations

    def test_rounded_citation_allowed(self, prop_pick):
        e = build_prop_evidence(prop_pick)
        # model_median = 28.4 cited as integer 28 → allowed (rounding)
        text = "LeBron James OVER 25.5: model says 28."
        assert verify_narrative(text, e).ok


# ---------------------------------------------------------------------------
# Templates (round-trip: template output must always verify)
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_prop_template_roundtrip(self, prop_pick, player_row, injuries):
        e = build_prop_evidence(prop_pick, player_row, injuries)
        text = template_narrative(e)
        result = verify_narrative(text, e)
        assert result.ok, (text, result.violations)

    def test_under_template_roundtrip(self, under_pick, player_row):
        e = build_prop_evidence(under_pick, player_row)
        text = template_narrative(e)
        assert "UNDER" in text
        result = verify_narrative(text, e)
        assert result.ok, (text, result.violations)

    def test_game_template_roundtrip(self, game_pick):
        e = build_game_evidence(game_pick)
        text = template_narrative(e)
        result = verify_narrative(text, e)
        assert result.ok, (text, result.violations)


# ---------------------------------------------------------------------------
# Generator (LLM mocked via subclass)
# ---------------------------------------------------------------------------

class _StubGenerator(NarrativeGenerator):
    """Generator with a canned LLM that never hits the network."""

    def __init__(self, responses: list[str]):
        super().__init__(api_key="stub-key")
        self._responses = list(responses)
        self.calls = 0

    def available(self) -> bool:
        return True

    def _call_llm(self, system: str, user: str) -> str:
        self.calls += 1
        return self._responses.pop(0)


class TestGenerator:
    def test_no_api_key_uses_template(self, prop_pick, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        gen = NarrativeGenerator(api_key="")
        e = build_prop_evidence(prop_pick)
        result = gen.generate(e)
        assert result.source == "template"
        assert verify_narrative(result.text, e).ok

    def test_valid_llm_text_accepted_first_try(self, prop_pick):
        good = "LeBron James OVER 25.5 points: model projects 28.4, a 61.2% shot with an 11.2pp edge."
        gen = _StubGenerator([good])
        result = gen.generate(build_prop_evidence(prop_pick))
        assert result.source == "llm"
        assert result.text == good
        assert gen.calls == 1

    def test_hallucination_retried_then_accepted(self, prop_pick):
        bad = "LeBron James OVER 25.5 — he scored 52 in the bubble."
        good = "LeBron James OVER 25.5 points: model projects 28.4 with a 61.2% chance."
        gen = _StubGenerator([bad, good])
        result = gen.generate(build_prop_evidence(prop_pick))
        assert result.source == "llm-retry"
        assert gen.calls == 2

    def test_double_hallucination_falls_back_to_template(self, prop_pick):
        bad = "LeBron James OVER 25.5 — he scored 52 in the bubble."
        gen = _StubGenerator([bad, bad])
        e = build_prop_evidence(prop_pick)
        result = gen.generate(e)
        assert result.source == "template"
        assert result.violations                      # audit trail retained
        assert verify_narrative(result.text, e).ok    # fallback is clean

    def test_llm_exception_falls_back_to_template(self, prop_pick):
        class _Boom(NarrativeGenerator):
            def available(self):
                return True

            def _call_llm(self, system, user):
                raise RuntimeError("network down")

        result = _Boom(api_key="x").generate(build_prop_evidence(prop_pick))
        assert result.source == "template"


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

class TestBatch:
    def test_narratives_for_picks_dedupes(self, prop_pick, game_pick, player_row):
        player_df = pd.DataFrame([player_row])
        gen = _StubGenerator([
            "LeBron James OVER 25.5 points: model projects 28.4.",
            "SAC to win: model gives 63% against an implied 52%.",
        ])
        results = narratives_for_picks(
            [prop_pick, prop_pick, game_pick],
            player_df=player_df,
            generator=gen,
        )
        assert len(results) == 2
        assert gen.calls == 2
        assert pick_key(prop_pick) in results
        assert pick_key(game_pick) in results

    def test_pick_key_stable(self, prop_pick, game_pick):
        assert pick_key(prop_pick) == "LeBron James|PTS|25.5|over"
        assert pick_key(game_pick) == "G2|home"

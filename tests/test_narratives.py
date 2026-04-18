from __future__ import annotations

import pytest

from src.narratives.token_budget import (
    BudgetResult,
    check_budget,
    count_tokens,
    count_words,
    enforce_word_limit,
)
from src.narratives.position_narrative import build_position_narrative
from src.narratives.candidate_narrative import build_candidate_narrative
from src.narratives.game_narrative import GameNarrative
from src.storage.models import (
    CandidateMoveRecord,
    PrimitiveValue,
    WeightedPrimitiveState,
)
from src.policies.profiles import load_policy


# ── token_budget ─────────────────────────────────────────────────────────────

class TestCountWords:
    def test_empty(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   ") == 0

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_multiple_words(self):
        assert count_words("the quick brown fox") == 4


class TestCountTokens:
    def test_returns_int(self):
        result = count_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world this is a longer sentence with many words")
        assert long > short


class TestEnforceWordLimit:
    def test_under_limit(self):
        text, truncated = enforce_word_limit("one two three", 10)
        assert text == "one two three"
        assert truncated is False

    def test_at_limit(self):
        text, truncated = enforce_word_limit("one two three", 3)
        assert text == "one two three"
        assert truncated is False

    def test_over_limit(self):
        text, truncated = enforce_word_limit("one two three four five", 3)
        assert text == "one two three"
        assert truncated is True


class TestCheckBudget:
    def test_returns_budget_result(self):
        result = check_budget("hello world", 50)
        assert isinstance(result, BudgetResult)

    def test_not_truncated(self):
        result = check_budget("short text", 100)
        assert result.was_truncated is False
        assert result.text == "short text"

    def test_truncated(self):
        text = " ".join(["word"] * 20)
        result = check_budget(text, 5)
        assert result.was_truncated is True
        assert count_words(result.text) == 5

    def test_word_count_matches(self):
        result = check_budget("one two three", 10)
        assert result.word_count == 3

    def test_token_count_positive(self):
        result = check_budget("hello world", 50)
        assert result.token_count > 0

    def test_budget_type_stored(self):
        result = check_budget("hello", 10, budget_type="words")
        assert result.budget_type == "words"


# ── position_narrative ────────────────────────────────────────────────────────

def _make_weighted_state(
    primitive_id: str,
    category: str,
    normalized_value: float,
    weight: float = 1.0,
    confidence: float = 1.0,
) -> WeightedPrimitiveState:
    weighted_score = normalized_value * weight * confidence
    if weighted_score >= 1.5:
        label = "critical"
    elif weighted_score >= 1.0:
        label = "high"
    elif weighted_score >= 0.5:
        label = "medium"
    elif weighted_score >= 0.2:
        label = "low"
    else:
        label = "negligible"
    return WeightedPrimitiveState(
        primitive_id=primitive_id,
        name=primitive_id.replace("_", " "),
        category=category,
        raw_value=normalized_value,
        normalized_value=normalized_value,
        weight=weight,
        confidence=confidence,
        weighted_score=weighted_score,
        importance_label=label,
        policy_message="",
    )


def _make_primitive(primitive_id: str, text_render: str, normalized_value: float = 0.8) -> PrimitiveValue:
    return PrimitiveValue(
        primitive_id=primitive_id,
        name=primitive_id.replace("_", " "),
        side="self",
        value=1,
        normalized_value=normalized_value,
        text_render=text_render,
        confidence=1.0,
        category="material",
    )


class TestBuildPositionNarrative:
    def test_returns_budget_result(self):
        policy = load_policy("balanced")
        states = [_make_weighted_state("self_material_difference", "material", 0.8)]
        prims = [_make_primitive("self_material_difference", "Material advantage.")]
        result = build_position_narrative(states, prims, policy)
        assert isinstance(result, BudgetResult)

    def test_includes_positive_signal(self):
        policy = load_policy("balanced")
        states = [_make_weighted_state("self_material_difference", "material", 0.9, weight=2.0)]
        prims = [_make_primitive("self_material_difference", "Strong material advantage.")]
        result = build_position_narrative(states, prims, policy)
        assert "Strong material advantage" in result.text

    def test_includes_negative_signal(self):
        policy = load_policy("balanced")
        states = [_make_weighted_state("self_hanging_own_pieces", "tactical", 0.1, weight=1.5)]
        prims = [_make_primitive("self_hanging_own_pieces", "Pieces are hanging.", normalized_value=0.1)]
        result = build_position_narrative(states, prims, policy)
        assert "Pieces are hanging" in result.text

    def test_respects_word_limit(self):
        policy = load_policy("balanced")
        states = [_make_weighted_state(f"self_p{i}", "material", 0.9, weight=2.0) for i in range(10)]
        prims = [_make_primitive(f"self_p{i}", " ".join(["word"] * 30)) for i in range(10)]
        result = build_position_narrative(states, prims, policy, word_limit=50)
        assert result.word_count <= 50

    def test_empty_inputs(self):
        policy = load_policy("balanced")
        result = build_position_narrative([], [], policy)
        assert isinstance(result, BudgetResult)

    def test_policy_directive_present(self):
        policy = load_policy("aggressive")
        result = build_position_narrative([], [], policy)
        assert "active play" in result.text.lower() or "initiative" in result.text.lower()


# ── candidate_narrative ───────────────────────────────────────────────────────

def _make_candidate(
    uci: str = "e2e4",
    san: str = "e4",
    engine_eval: float | None = None,
    risk_flags: list[str] | None = None,
    delta_summary: str = "No significant primitive changes.",
) -> CandidateMoveRecord:
    return CandidateMoveRecord(
        uci=uci,
        san=san,
        source="engine",
        engine_eval_before=None,
        engine_eval_after=engine_eval,
        risk_flags=risk_flags or [],
        primitive_delta_summary=delta_summary,
        primitives_after=[],
    )


class TestBuildCandidateNarrative:
    def test_returns_budget_result(self):
        c = _make_candidate()
        result = build_candidate_narrative(c)
        assert isinstance(result, BudgetResult)

    def test_high_eval_positive_message(self):
        c = _make_candidate(engine_eval=150.0)
        result = build_candidate_narrative(c)
        assert "strongly favors" in result.text.lower()

    def test_negative_eval_message(self):
        c = _make_candidate(engine_eval=-150.0)
        result = build_candidate_narrative(c)
        assert "poorly" in result.text.lower()

    def test_risk_flag_included(self):
        c = _make_candidate(risk_flags=["hangs_piece"])
        result = build_candidate_narrative(c)
        assert "hanging" in result.text.lower()

    def test_multiple_risk_flags(self):
        c = _make_candidate(risk_flags=["hangs_piece", "exposes_king"])
        result = build_candidate_narrative(c)
        assert "hanging" in result.text.lower()
        assert "king" in result.text.lower()

    def test_delta_summary_included(self):
        c = _make_candidate(delta_summary="+: center occupancy (+0.20)")
        result = build_candidate_narrative(c)
        assert "center occupancy" in result.text

    def test_respects_word_limit(self):
        c = _make_candidate(
            engine_eval=200.0,
            risk_flags=["hangs_piece", "exposes_king", "loses_material"],
            delta_summary="+: " + ", ".join([f"feature{i} (+0.10)" for i in range(20)]),
        )
        result = build_candidate_narrative(c, word_limit=20)
        assert result.word_count <= 20

    def test_no_eval_no_flags(self):
        c = _make_candidate()
        result = build_candidate_narrative(c)
        assert result.text  # not empty


# ── game_narrative ────────────────────────────────────────────────────────────

class TestGameNarrative:
    def test_add_and_render(self):
        gn = GameNarrative(max_words=500)
        gn.add(1, "e4", "Opened with king's pawn.")
        rendered = gn.render()
        assert "e4" in rendered
        assert "Opened with king's pawn" in rendered

    def test_multiple_entries(self):
        gn = GameNarrative(max_words=500)
        gn.add(1, "e4", "King's pawn.")
        gn.add(2, "e5", "Symmetric response.")
        rendered = gn.render()
        assert "e4" in rendered
        assert "e5" in rendered

    def test_prune_on_overflow(self):
        gn = GameNarrative(max_words=10)
        for i in range(20):
            gn.add(i, f"m{i}", " ".join(["word"] * 5))
        # After pruning, rendered text should be within budget
        assert count_words(gn.render()) <= 10 or len(gn._entries) == 1  # at least 1 entry kept

    def test_to_dict_structure(self):
        gn = GameNarrative()
        gn.add(1, "e4", "Summary.")
        d = gn.to_dict()
        assert "max_words" in d
        assert "entries" in d
        assert "rendered" in d
        assert "word_count" in d

    def test_empty_render(self):
        gn = GameNarrative()
        assert gn.render() == ""

    def test_word_count_in_to_dict(self):
        gn = GameNarrative()
        gn.add(1, "Nf3", "Knight development.")
        d = gn.to_dict()
        assert d["word_count"] == count_words(d["rendered"])

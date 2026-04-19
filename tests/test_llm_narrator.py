"""Tests for the v1.2 interpreter pipeline."""
from __future__ import annotations

import json
import re

import pytest

from src.llm.prompt_version import CURRENT_VERSION, INTERPRETER_VERSIONS, VERSIONS, get_interpreter_template_path
from src.narratives.llm_narrator import (
    _build_interpreter_user_prompt,
    _game_phase,
    _parse_interpreter_response,
    _tone_label,
    build_interpreter_narrative,
)
from src.narratives.token_budget import (
    COMBINED_TARGET_TOKENS,
    DECISION_MAX_TOKENS,
    INTERPRETER_MAX_TOKENS,
)
from src.policies.profiles import PolicyProfile
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionRecord,
    InterpreterNarrative,
    MoveTrace,
    PrimitiveValue,
    WeightedPrimitiveState,
)
from src.llm.prompts import (
    _build_v12_candidate_block,
    _RISK_PROSE,
    build_prompt,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _policy() -> PolicyProfile:
    return PolicyProfile(
        policy_id="balanced",
        name="Balanced",
        description="Moderate emphasis.",
        plain_language="Weigh all factors equally. No single concern dominates.",
        weight_map={"material_difference": 2.0, "castled_status": 1.5},
        group_weights={},
    )


def _weighted_state(pid: str, category: str, score: float) -> WeightedPrimitiveState:
    return WeightedPrimitiveState(
        primitive_id=pid, name=pid, category=category,
        raw_value=score, normalized_value=score,
        weight=1.0, confidence=1.0, weighted_score=score,
        importance_label="medium", policy_message="",
    )


def _primitive(pid: str, text: str) -> PrimitiveValue:
    return PrimitiveValue(
        primitive_id=pid, name=pid, side="self",
        value=0.5, normalized_value=0.5,
        text_render=text, confidence=1.0, category="material",
    )


def _snapshot() -> BoardSnapshot:
    return BoardSnapshot(
        game_id="t", ply_index=0,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        turn="white", legal_moves=["e2e4"], move_history=[],
        is_check=False, is_game_over=False,
    )


def _candidate(uci: str, san: str, rank: int = 1, pidx: int = 0,
               score_delta: float | None = None,
               delta_summary: str = "",
               risk_flags: list[str] | None = None) -> CandidateMoveRecord:
    return CandidateMoveRecord(
        uci=uci, san=san, internal_rank=rank, presentation_index=pidx,
        source="engine", risk_flags=risk_flags or [],
        score_delta=score_delta, primitive_delta_summary=delta_summary,
    )


# ── tone label ────────────────────────────────────────────────────────────────

class TestToneLabel:
    def test_critical(self):      assert _tone_label(1.5) == "critical"
    def test_high(self):          assert _tone_label(1.0) == "high"
    def test_moderate(self):      assert _tone_label(0.5) == "moderate"
    def test_low(self):           assert _tone_label(0.1) == "low"
    def test_boundary_critical(self): assert _tone_label(1.4) == "critical"
    def test_boundary_high(self):     assert _tone_label(0.8) == "high"
    def test_boundary_moderate(self): assert _tone_label(0.3) == "moderate"


# ── game phase ────────────────────────────────────────────────────────────────

class TestGamePhase:
    def test_opening(self):    assert _game_phase(1) == "opening"
    def test_opening_max(self): assert _game_phase(10) == "opening"
    def test_middlegame(self):  assert _game_phase(11) == "middlegame"
    def test_middlegame_max(self): assert _game_phase(25) == "middlegame"
    def test_endgame(self):    assert _game_phase(26) == "endgame"


# ── interpreter prompt — no numeric values ────────────────────────────────────

class TestInterpreterPromptNoNumbers:
    def _build(self) -> str:
        ws = [
            _weighted_state("self_material_difference", "material", 0.9),
            _weighted_state("self_castled_status", "king_safety", 0.5),
        ]
        prims = [
            _primitive("self_material_difference", "We have a material advantage."),
            _primitive("self_castled_status", "King is not yet castled."),
        ]
        return _build_interpreter_user_prompt(ws, prims, _policy(), ["e4", "e5"], 2)

    def test_no_numeric_values(self):
        prompt = self._build()
        assert not re.search(r"\b\d+\.?\d*\b", prompt), \
            f"Found numeric values in interpreter prompt:\n{prompt}"

    def test_no_primitive_ids(self):
        prompt = self._build()
        assert "self_material_difference" not in prompt
        assert "self_castled_status" not in prompt

    def test_contains_tone_labels(self):
        prompt = self._build()
        assert any(t in prompt for t in ("critical", "high", "moderate", "low"))

    def test_contains_text_renders(self):
        prompt = self._build()
        assert "material advantage" in prompt

    def test_contains_policy_plain_language(self):
        prompt = self._build()
        assert "Weigh all factors equally" in prompt

    def test_contains_game_phase(self):
        prompt = self._build()
        assert "opening" in prompt


# ── parse interpreter response ────────────────────────────────────────────────

_GOOD_RESPONSE = """SITUATION: This is a balanced open position.
ALERT: None
PRIORITY: Complete development and protect the king.
DIRECTIVE: A developing move is most appropriate here."""


class TestParseInterpreterResponse:
    def test_all_four_sections_present(self):
        result = _parse_interpreter_response(_GOOD_RESPONSE)
        assert "situation" in result
        assert "alert" in result
        assert "priority" in result
        assert "directive" in result

    def test_values_populated(self):
        result = _parse_interpreter_response(_GOOD_RESPONSE)
        assert result["situation"] == "This is a balanced open position."
        assert result["alert"] == "None"
        assert "development" in result["priority"]
        assert "developing" in result["directive"]

    def test_missing_section_returns_unavailable(self):
        partial = "SITUATION: Some position.\nPRIORITY: Develop pieces.\nDIRECTIVE: Play quiet."
        result = _parse_interpreter_response(partial)
        assert result["alert"] == "unavailable"

    def test_empty_string_returns_all_unavailable(self):
        result = _parse_interpreter_response("")
        for key in ("situation", "alert", "priority", "directive"):
            assert result[key] == "unavailable"


# ── build_interpreter_narrative — mock LLM ───────────────────────────────────

class TestBuildInterpreterNarrative:
    def _mock_call(self, sys, usr) -> str:
        return _GOOD_RESPONSE

    def _mock_fail(self, sys, usr) -> str:
        raise RuntimeError("LLM unavailable")

    def _ws_and_prims(self):
        ws = [_weighted_state("self_material_difference", "material", 0.9)]
        prims = [_primitive("self_material_difference", "We have a material advantage.")]
        return ws, prims

    def test_returns_interpreter_narrative(self):
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), ["e4"], 2, self._mock_call)
        assert isinstance(result, InterpreterNarrative)

    def test_all_four_sections_in_result(self):
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), [], 1, self._mock_call)
        assert result.situation != ""
        assert result.alert != ""
        assert result.priority != ""
        assert result.directive != ""

    def test_prompt_version_set(self):
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), [], 1, self._mock_call)
        assert result.prompt_version == "interpreter_v1.0"

    def test_token_count_positive(self):
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), [], 1, self._mock_call)
        assert result.token_count > 0

    def test_llm_failure_sets_unavailable_no_raise(self):
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), [], 1, self._mock_fail)
        assert result.situation == "unavailable"
        assert result.alert == "unavailable"
        assert result.token_count == 0

    def test_parse_failure_sets_unavailable_no_raise(self):
        def bad_output(sys, usr): return "completely garbled output no sections"
        ws, prims = self._ws_and_prims()
        result = build_interpreter_narrative(ws, prims, _policy(), [], 1, bad_output)
        for field in ("situation", "alert", "priority", "directive"):
            assert getattr(result, field) == "unavailable"


# ── v1.2 decision prompt — no FEN, no source labels ──────────────────────────

class TestV12DecisionPrompt:
    def _interp(self) -> InterpreterNarrative:
        return InterpreterNarrative(
            situation="Balanced position.",
            alert="None",
            priority="Develop pieces.",
            directive="Play a developing move.",
            raw_output=_GOOD_RESPONSE,
            token_count=50,
        )

    def _candidates(self) -> list[CandidateMoveRecord]:
        return [
            _candidate("e2e4", "e4", rank=1, pidx=0, score_delta=0.5,
                        delta_summary="+: center occupancy (+0.25)"),
            _candidate("g1f3", "Nf3", rank=2, pidx=1, score_delta=0.3,
                        delta_summary="+: minor pieces developed (+0.30)"),
        ]

    def test_no_fen_in_v12_prompt(self):
        snap = _snapshot()
        pr = build_prompt(snap, "", "", self._candidates(), version="v1.2",
                          interpreter_narrative=self._interp())
        assert "FEN:" not in pr.user_prompt
        assert snap.fen not in pr.user_prompt

    def test_no_source_labels_in_candidate_block(self):
        snap = _snapshot()
        pr = build_prompt(snap, "", "", self._candidates(), version="v1.2",
                          interpreter_narrative=self._interp())
        for label in ("engine", "heuristic", "random", "hybrid"):
            assert label not in pr.candidate_block

    def test_interpreter_sections_appear_in_prompt(self):
        snap = _snapshot()
        interp = self._interp()
        pr = build_prompt(snap, "", "", self._candidates(), version="v1.2",
                          interpreter_narrative=interp)
        assert "SITUATION:" in pr.user_prompt
        assert "ALERT:" in pr.user_prompt
        assert "PRIORITY:" in pr.user_prompt
        assert "DIRECTIVE:" in pr.user_prompt

    def test_policy_plain_language_in_prompt(self):
        snap = _snapshot()
        pr = build_prompt(snap, "", "", self._candidates(), version="v1.2",
                          interpreter_narrative=self._interp(),
                          policy_plain_language="Weigh all factors equally.")
        assert "Weigh all factors equally." in pr.user_prompt

    def test_v11_still_has_fen(self):
        snap = _snapshot()
        pr = build_prompt(snap, "Narrative.", "Policy.", self._candidates(), version="v1.1")
        assert "FEN:" in pr.user_prompt
        assert snap.fen in pr.user_prompt


# ── v1.2 candidate block — tone vocab, no numbers ────────────────────────────

class TestV12CandidateBlock:
    def test_no_numeric_values_in_block(self):
        candidates = [
            _candidate("e2e4", "e4", score_delta=0.8,
                        delta_summary="+: center occupancy (+0.25), bishop activity (+0.15)"),
            _candidate("g1f3", "Nf3", score_delta=0.3,
                        delta_summary="+: minor pieces developed (+0.30)"),
            _candidate("d2d4", "d4", score_delta=0.1,
                        delta_summary="+: center pawn presence (+0.05)"),
        ]
        block = _build_v12_candidate_block(candidates)
        # strip the UCI (which looks like letters and digits)
        no_uci = re.sub(r'\([a-h][1-8][a-h][1-8]\)', '', block)
        assert not re.search(r'\b\d+\.?\d*\b', no_uci), \
            f"Numeric values found in candidate block:\n{block}"

    def test_tone_strong_for_top_delta(self):
        candidates = [
            _candidate("e2e4", "e4", score_delta=0.9,
                        delta_summary="+: center occupancy (+0.25)"),
            _candidate("g1f3", "Nf3", score_delta=0.1,
                        delta_summary="+: minor pieces developed (+0.05)"),
        ]
        block = _build_v12_candidate_block(candidates)
        assert "Strong" in block

    def test_tone_modest_for_small_delta(self):
        candidates = [
            _candidate("e2e4", "e4", score_delta=0.9,
                        delta_summary="+: center occupancy (+0.25)"),
            _candidate("g1f3", "Nf3", score_delta=0.1,
                        delta_summary="+: minor pieces developed (+0.05)"),
        ]
        block = _build_v12_candidate_block(candidates)
        assert "Modest" in block

    def test_no_delta_neutral_label(self):
        candidates = [_candidate("e2e4", "e4", score_delta=None)]
        block = _build_v12_candidate_block(candidates)
        assert "neutral" in block.lower() or "Positionally" in block

    def test_risk_hangs_piece_prose(self):
        candidates = [_candidate("e2e4", "e4", score_delta=0.5, risk_flags=["hangs_piece"])]
        block = _build_v12_candidate_block(candidates)
        assert "leaves a piece exposed" in block

    def test_risk_exposes_king_prose(self):
        candidates = [_candidate("e2e4", "e4", score_delta=0.5, risk_flags=["exposes_king"])]
        block = _build_v12_candidate_block(candidates)
        assert "weakens king safety" in block

    def test_risk_loses_material_prose(self):
        candidates = [_candidate("e2e4", "e4", score_delta=0.5, risk_flags=["loses_material"])]
        block = _build_v12_candidate_block(candidates)
        assert "risks material loss" in block

    def test_risk_king_under_pressure_prose(self):
        candidates = [_candidate("e2e4", "e4", score_delta=0.5, risk_flags=["king_under_pressure"])]
        block = _build_v12_candidate_block(candidates)
        assert "increases pressure on your king" in block

    def test_no_risk_flags_shows_no_apparent_risk(self):
        candidates = [_candidate("e2e4", "e4", score_delta=0.5)]
        block = _build_v12_candidate_block(candidates)
        assert "No apparent risk" in block


# ── MoveTrace stores new fields ───────────────────────────────────────────────

class TestMoveTraceInterpreterFields:
    def _interp(self) -> InterpreterNarrative:
        return InterpreterNarrative(
            situation="Balanced.", alert="None",
            priority="Develop.", directive="Play quiet.",
            raw_output="...", token_count=85,
        )

    def test_interpreter_narrative_stored(self):
        snap = _snapshot()
        interp = self._interp()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap,
                          interpreter_narrative=interp)
        assert trace.interpreter_narrative is not None
        assert trace.interpreter_narrative.situation == "Balanced."

    def test_interpreter_tokens_stored(self):
        snap = _snapshot()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap,
                          interpreter_tokens=85)
        assert trace.interpreter_tokens == 85

    def test_total_tokens_stored(self):
        snap = _snapshot()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap,
                          interpreter_tokens=85, total_tokens_this_move=985)
        assert trace.total_tokens_this_move == 985

    def test_combined_token_count_arithmetic(self):
        interp_tok = 120
        decision_tok = 850
        snap = _snapshot()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap,
                          interpreter_tokens=interp_tok,
                          total_tokens_this_move=interp_tok + decision_tok)
        assert trace.total_tokens_this_move == interp_tok + decision_tok

    def test_defaults_are_none(self):
        snap = _snapshot()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap)
        assert trace.interpreter_narrative is None
        assert trace.interpreter_tokens is None
        assert trace.total_tokens_this_move is None

    def test_to_json_includes_interpreter_fields(self):
        snap = _snapshot()
        interp = self._interp()
        trace = MoveTrace(game_id="g", ply_index=0, board_snapshot=snap,
                          interpreter_narrative=interp, interpreter_tokens=85,
                          total_tokens_this_move=985)
        data = json.loads(trace.to_json())
        assert data["interpreter_tokens"] == 85
        assert data["total_tokens_this_move"] == 985
        assert data["interpreter_narrative"]["situation"] == "Balanced."


# ── prompt version registry ───────────────────────────────────────────────────

class TestPromptVersionRegistry:
    def test_v12_in_versions(self):
        assert "v1.2" in VERSIONS

    def test_v12_template_file_exists(self):
        import os
        path = VERSIONS["v1.2"]
        assert os.path.exists(path), f"v1.2 template not found at {path}"

    def test_interpreter_v10_in_interpreter_versions(self):
        assert "interpreter_v1.0" in INTERPRETER_VERSIONS

    def test_interpreter_template_file_exists(self):
        import os
        path = INTERPRETER_VERSIONS["interpreter_v1.0"]
        assert os.path.exists(path), f"interpreter_v1.0 template not found at {path}"

    def test_v12_registered_in_versions(self):
        assert "v1.2" in VERSIONS

    def test_get_interpreter_template_path_returns_string(self):
        path = get_interpreter_template_path()
        assert isinstance(path, str)
        assert path.endswith(".md")

    def test_unknown_interpreter_version_raises(self):
        with pytest.raises(ValueError):
            get_interpreter_template_path("interpreter_v99.0")


# ── token budget constants ────────────────────────────────────────────────────

class TestTokenBudgetConstants:
    def test_interpreter_max_tokens(self):
        assert INTERPRETER_MAX_TOKENS == 400

    def test_decision_max_tokens(self):
        assert DECISION_MAX_TOKENS == 1000

    def test_combined_target_tokens(self):
        assert COMBINED_TARGET_TOKENS == 1400

    def test_combined_equals_sum(self):
        assert COMBINED_TARGET_TOKENS == INTERPRETER_MAX_TOKENS + DECISION_MAX_TOKENS

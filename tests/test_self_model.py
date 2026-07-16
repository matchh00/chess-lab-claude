"""Tests for Phase 9 (self-model experiment): memory, pattern detection,
renderers, prompt integration, calibration analytics, and configs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import yaml

import src.llm.prompts as prompts_module
from src.analytics.self_model_metrics import (
    compute_confidence_calibration,
    compute_self_model_usage,
)
from src.llm.prompt_version import VERSIONS, get_template_path
from src.llm.prompts import build_prompt
from src.self_model.memory import SelfModelMemory, dominant_categories
from src.self_model.models import MoveReflection, SelfModelState
from src.self_model.renderers import render_history_block, render_self_model_block
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionRecord,
    MoveTrace,
    WeightedPrimitiveState,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reflection(
    ply: int = 0,
    san: str = "e4",
    confidence: float = 0.5,
    cpl: Optional[float] = 20.0,
    blunder_label: Optional[str] = None,
    categories: Optional[list[str]] = None,
    fallback: bool = False,
) -> MoveReflection:
    return MoveReflection(
        game_id="test-game",
        ply_index=ply,
        san=san,
        uci="e2e4",
        confidence=confidence,
        centipawn_loss=cpl,
        blunder_label=blunder_label,
        reasoning_summary="test reasoning",
        dominant_categories=categories or ["center"],
        fallback_used=fallback,
    )


def make_weighted_state(category: str, score: float) -> WeightedPrimitiveState:
    return WeightedPrimitiveState(
        primitive_id=f"self_{category}_test",
        name=f"{category} test",
        category=category,
        raw_value=1,
        normalized_value=0.5,
        weight=1.0,
        confidence=1.0,
        weighted_score=score,
        importance_label="medium",
        policy_message="",
    )


def make_snapshot() -> BoardSnapshot:
    return BoardSnapshot(
        game_id="test-game",
        ply_index=0,
        fen=STARTING_FEN,
        turn="white",
        legal_moves=["e2e4", "d2d4"],
        move_history=[],
        is_check=False,
        is_game_over=False,
    )


def make_candidate(uci: str = "e2e4", san: str = "e4", idx: int = 0) -> CandidateMoveRecord:
    return CandidateMoveRecord(
        uci=uci, san=san, internal_rank=idx + 1, presentation_index=idx, source="engine",
    )


def make_move_trace(
    confidence: float,
    cpl: Optional[float],
    fallback: bool = False,
    self_model_mode: str = "off",
    self_model_block: str = "",
) -> MoveTrace:
    return MoveTrace(
        game_id="test-game",
        ply_index=0,
        board_snapshot=make_snapshot(),
        decision_record=DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=1,
            selected_presentation_index=0,
            confidence=confidence,
            fallback_used=fallback,
            is_valid=not fallback,
        ),
        centipawn_loss=cpl,
        self_model_mode=self_model_mode,
        self_model_block=self_model_block,
    )


@pytest.fixture(autouse=True)
def stub_token_counter(monkeypatch):
    """count_tokens needs a network fetch for its encoding; stub it for unit tests."""
    monkeypatch.setattr(prompts_module, "count_tokens", lambda text: len(text.split()))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestSelfModelMemory:
    def test_empty_state(self):
        state = SelfModelMemory().state()
        assert state.moves_observed == 0
        assert state.mean_confidence is None
        assert state.mean_cpl is None
        assert state.patterns == []

    def test_observe_accumulates(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(ply=0, confidence=0.4, cpl=10.0))
        mem.observe(make_reflection(ply=2, confidence=0.6, cpl=30.0))
        state = mem.state()
        assert state.moves_observed == 2
        assert state.mean_confidence == pytest.approx(0.5)
        assert state.mean_cpl == pytest.approx(20.0)

    def test_game_scope_resets_between_games(self):
        mem = SelfModelMemory(scope="game")
        mem.observe(make_reflection())
        mem.start_game()
        assert mem.state().moves_observed == 0

    def test_run_scope_persists_between_games(self):
        mem = SelfModelMemory(scope="run")
        mem.observe(make_reflection())
        mem.start_game()
        assert mem.state().moves_observed == 1

    def test_fallback_moves_excluded_from_confidence(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.8, cpl=10.0))
        mem.observe(make_reflection(confidence=0.0, cpl=300.0, fallback=True))
        state = mem.state()
        assert state.mean_confidence == pytest.approx(0.8)
        # but the fallback move's CPL still counts toward outcomes
        assert state.mean_cpl == pytest.approx(155.0)

    def test_observe_move_builds_reflection(self):
        mem = SelfModelMemory()
        decision = DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=1,
            selected_presentation_index=2,
            reasoning_summary="controls the center",
            confidence=0.7,
        )
        states = [make_weighted_state("initiative", 0.9), make_weighted_state("center", 0.5)]
        reflection = mem.observe_move(
            game_id="g", ply_index=4, san="e4", decision=decision,
            centipawn_loss=42.0, blunder_label=None, weighted_states=states,
        )
        assert reflection.dominant_categories == ["initiative", "center"]
        assert reflection.confidence == 0.7
        assert mem.state().moves_observed == 1

    def test_dominant_categories_dedupes(self):
        states = [
            make_weighted_state("initiative", 0.9),
            make_weighted_state("initiative", 0.8),
            make_weighted_state("king_safety", 0.7),
        ]
        assert dominant_categories(states) == ["initiative", "king_safety"]


class TestPatternDetection:
    def test_overconfidence_pattern(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(ply=0, san="Nf3", confidence=0.8, cpl=150.0))
        mem.observe(make_reflection(ply=2, san="Bc4", confidence=0.9, cpl=250.0, blunder_label="blunder"))
        kinds = [p.kind for p in mem.state().patterns]
        assert "overconfidence" in kinds

    def test_no_overconfidence_pattern_below_threshold(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.8, cpl=150.0))
        mem.observe(make_reflection(confidence=0.3, cpl=200.0))
        kinds = [p.kind for p in mem.state().patterns]
        assert "overconfidence" not in kinds

    def test_calibration_gap_pattern(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.9, cpl=150.0))  # costly, high conf
        mem.observe(make_reflection(confidence=0.5, cpl=10.0))   # clean, lower conf
        kinds = [p.kind for p in mem.state().patterns]
        assert "calibration_gap" in kinds

    def test_category_bias_pattern(self):
        mem = SelfModelMemory()
        # initiative-dominated decisions are costly; center-dominated are clean
        for ply, cpl in [(0, 200.0), (2, 250.0), (4, 300.0)]:
            mem.observe(make_reflection(ply=ply, cpl=cpl, confidence=0.4,
                                        categories=["initiative"]))
        for ply in (6, 8, 10):
            mem.observe(make_reflection(ply=ply, cpl=5.0, confidence=0.4,
                                        categories=["center"]))
        patterns = mem.state().patterns
        bias = [p for p in patterns if p.kind == "category_bias"]
        assert len(bias) == 1
        assert "initiative" in bias[0].text

    def test_cost_streak_pattern(self):
        mem = SelfModelMemory()
        for ply in (0, 2, 4):
            mem.observe(make_reflection(ply=ply, cpl=120.0, confidence=0.4))
        kinds = [p.kind for p in mem.state().patterns]
        assert "cost_streak" in kinds

    def test_clean_play_produces_no_patterns(self):
        mem = SelfModelMemory()
        for ply in (0, 2, 4, 6):
            mem.observe(make_reflection(ply=ply, cpl=15.0, confidence=0.6))
        assert mem.state().patterns == []


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

class TestRenderers:
    def test_empty_state_renders_nothing(self):
        state = SelfModelState()
        assert render_history_block(state) == ""
        assert render_self_model_block(state) == ""

    def test_history_block_lists_moves(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(ply=0, san="e4", confidence=0.8, cpl=20.0))
        mem.observe(make_reflection(ply=2, san="Nf3", confidence=0.6, cpl=250.0,
                                    blunder_label="blunder"))
        block = render_history_block(mem.state())
        assert block.startswith("## Decision History")
        assert "e4" in block and "Nf3" in block
        assert "blunder" in block
        assert "0.80" in block

    def test_history_block_caps_move_count(self):
        mem = SelfModelMemory()
        for ply in range(0, 20, 2):
            mem.observe(make_reflection(ply=ply, san=f"m{ply}", cpl=10.0))
        block = render_history_block(mem.state(), max_moves=4)
        assert block.count("- ply") == 4
        assert "ply 18" in block  # most recent kept
        assert "ply 0:" not in block

    def test_history_block_is_factual_only(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.9, cpl=300.0, blunder_label="blunder"))
        mem.observe(make_reflection(confidence=0.9, cpl=300.0, blunder_label="blunder"))
        block = render_history_block(mem.state())
        assert "Directives" not in block
        assert "Known biases" not in block

    def test_self_model_block_includes_patterns_and_directives(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(ply=0, confidence=0.8, cpl=150.0))
        mem.observe(make_reflection(ply=2, confidence=0.9, cpl=250.0, blunder_label="blunder"))
        block = render_self_model_block(mem.state())
        assert block.startswith("## Self-Model")
        assert "### Recent decisions" in block
        assert "### Known biases" in block
        assert "### Directives" in block
        assert "confidence" in block.lower()

    def test_self_model_block_without_patterns_notes_clean_record(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(cpl=10.0, confidence=0.6))
        block = render_self_model_block(mem.state())
        assert "No recurring bias detected yet" in block

    def test_fallback_moves_flagged_in_history(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.0, cpl=300.0, fallback=True))
        block = render_history_block(mem.state())
        assert "fallback" in block


# ---------------------------------------------------------------------------
# Prompt integration
# ---------------------------------------------------------------------------

class TestPromptIntegration:
    def test_v12_template_registered(self):
        assert "v1.2" in VERSIONS
        assert Path(get_template_path("v1.2")).exists()

    def test_prompt_without_block_has_no_self_section(self):
        record = build_prompt(make_snapshot(), "narrative", "policy",
                              [make_candidate()], version="v1.2")
        assert "Self-Model" not in record.user_prompt
        assert "Decision History" not in record.user_prompt
        assert record.self_model_block == ""

    def test_prompt_with_block_includes_section_before_candidates(self):
        block = "## Self-Model (your own decision record and known biases)\n- test line"
        record = build_prompt(make_snapshot(), "narrative", "policy",
                              [make_candidate()], version="v1.2",
                              self_model_block=block)
        assert block in record.user_prompt
        assert record.user_prompt.index("## Self-Model") < record.user_prompt.index("## Candidate Moves")
        assert record.self_model_block == block

    def test_v12_system_prompt_mentions_self_model(self):
        record = build_prompt(make_snapshot(), "narrative", "policy",
                              [make_candidate()], version="v1.2")
        assert "Self-Model" in record.system_prompt

    def test_default_version_unchanged(self):
        record = build_prompt(make_snapshot(), "narrative", "policy", [make_candidate()])
        assert record.prompt_version == "v1.1"


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    def test_too_few_decisions(self):
        result = compute_confidence_calibration([make_move_trace(0.5, 10.0)])
        assert result["n_decisions"] == 1
        assert result["confidence_cpl_correlation"] is None

    def test_negative_correlation_for_calibrated_player(self):
        traces = [
            make_move_trace(0.9, 10.0),
            make_move_trace(0.8, 30.0),
            make_move_trace(0.4, 200.0),
            make_move_trace(0.3, 300.0),
        ]
        result = compute_confidence_calibration(traces)
        assert result["n_decisions"] == 4
        assert result["confidence_cpl_correlation"] < 0

    def test_fallback_decisions_excluded(self):
        traces = [
            make_move_trace(0.9, 10.0),
            make_move_trace(0.5, 50.0),
            make_move_trace(0.0, 500.0, fallback=True),
        ]
        result = compute_confidence_calibration(traces)
        assert result["n_decisions"] == 2

    def test_blunder_and_clean_confidence_split(self):
        traces = [
            make_move_trace(0.9, 250.0),  # blunder (>200)
            make_move_trace(0.4, 10.0),   # clean (<=50)
        ]
        result = compute_confidence_calibration(traces)
        assert result["mean_confidence_on_blunders"] == pytest.approx(0.9)
        assert result["mean_confidence_on_clean_moves"] == pytest.approx(0.4)

    def test_bins_cover_all_decisions(self):
        traces = [make_move_trace(c, 50.0) for c in (0.1, 0.55, 0.75, 1.0)]
        result = compute_confidence_calibration(traces)
        assert sum(b["n"] for b in result["bins"]) == 4

    def test_self_model_usage(self):
        traces = [
            make_move_trace(0.5, 10.0, self_model_mode="full"),  # first move: no block yet
            make_move_trace(0.5, 10.0, self_model_mode="full", self_model_block="## Self-Model\n- x"),
        ]
        usage = compute_self_model_usage(traces)
        assert usage["moves_with_self_model_block"] == 1
        assert usage["total_lab_moves"] == 2
        assert usage["modes_present"] == ["full"]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class TestExperimentConfigs:
    @pytest.mark.parametrize("name,mode", [
        ("self_model_a", "off"),
        ("self_model_b", "history"),
        ("self_model_c", "full"),
    ])
    def test_condition_configs(self, name, mode):
        with open(f"configs/experiments/{name}.yaml") as f:
            config = yaml.safe_load(f)
        assert config["config_id"] == name
        assert str(config["self_model_mode"]) == mode
        assert config["prompt_version"] == "v1.2"
        # conditions must be identical except for the self-model mode
        assert config["policy"] == "balanced"
        assert config["candidate_mode"] == "engine_assisted"
        assert config["narrative_mode"] == "on"
        assert config["random_seed"] == 42

    def test_state_round_trips_through_json(self):
        mem = SelfModelMemory()
        mem.observe(make_reflection(confidence=0.8, cpl=150.0))
        mem.observe(make_reflection(confidence=0.9, cpl=250.0, blunder_label="blunder"))
        state = mem.state()
        restored = SelfModelState.from_json(state.to_json())
        assert restored == state

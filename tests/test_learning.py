"""Tests for the learning module (src/learning/)."""
from __future__ import annotations

import copy

import pytest

from src.learning.evaluator import MoveEvaluation, evaluate_move
from src.learning.influence import InfluenceRecord, PrimitiveInfluenceEntry, _bare_key, score_move_influence
from src.learning.learning_log import GameAdjustmentEntry, LearningLog
from src.learning.rebalancer import WeightAdjustment, rebalance_weights, save_learned_policy
from src.learning.summary import (
    PerformanceRow,
    build_game_summary,
    build_primitive_summaries,
)
from src.policies.profiles import PolicyProfile
from src.storage.models import BoardSnapshot, DecisionRecord, MoveTrace, WeightedPrimitiveState


# ── fixtures ───────────────────────────────────────────────────────────────────

def _snapshot() -> BoardSnapshot:
    return BoardSnapshot(
        game_id="test",
        ply_index=0,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        turn="white",
        legal_moves=["e2e4"],
        move_history=[],
        is_check=False,
        is_game_over=False,
    )


def _weighted_state(primitive_id: str, category: str, score: float) -> WeightedPrimitiveState:
    return WeightedPrimitiveState(
        primitive_id=primitive_id,
        name=primitive_id,
        category=category,
        raw_value=score,
        normalized_value=score,
        weight=1.0,
        confidence=1.0,
        weighted_score=score,
        importance_label="medium",
        policy_message="",
    )


def _trace(
    cpl: float | None = None,
    selected_rank: int | None = None,
    weighted_states: list[WeightedPrimitiveState] | None = None,
    blunder_label: str | None = None,
) -> MoveTrace:
    dec = None
    if selected_rank is not None:
        dec = DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=selected_rank,
            selected_presentation_index=0,
        )
    return MoveTrace(
        game_id="test",
        ply_index=0,
        board_snapshot=_snapshot(),
        centipawn_loss=cpl,
        decision_record=dec,
        weighted_state=weighted_states or [],
        blunder_label=blunder_label,
    )


def _policy(weight_map: dict[str, float] | None = None) -> PolicyProfile:
    return PolicyProfile(
        policy_id="test",
        name="Test",
        description="",
        weight_map=weight_map or {"material_difference": 2.0, "castled_status": 1.5},
        group_weights={},
    )


# ── evaluator ──────────────────────────────────────────────────────────────────

class TestEvaluateMove:
    def test_good_move(self):
        ev = evaluate_move(_trace(cpl=10.0, selected_rank=1))
        assert ev.quality_label == "good"
        assert ev.rank_1_selected is True

    def test_inaccuracy(self):
        ev = evaluate_move(_trace(cpl=40.0))
        assert ev.quality_label == "inaccuracy"

    def test_mistake(self):
        ev = evaluate_move(_trace(cpl=75.0))
        assert ev.quality_label == "mistake"

    def test_blunder(self):
        ev = evaluate_move(_trace(cpl=200.0))
        assert ev.quality_label == "blunder"

    def test_no_cpl(self):
        ev = evaluate_move(_trace(cpl=None))
        assert ev.quality_label is None
        assert ev.centipawn_loss is None

    def test_no_decision(self):
        ev = evaluate_move(_trace(cpl=10.0))
        assert ev.rank_1_selected is False
        assert ev.selected_rank is None

    def test_rank_2_not_rank_1(self):
        ev = evaluate_move(_trace(cpl=10.0, selected_rank=2))
        assert ev.rank_1_selected is False


# ── bare_key ──────────────────────────────────────────────────────────────────

class TestBareKey:
    def test_self_prefix(self):
        assert _bare_key("self_material_difference") == "material_difference"

    def test_opponent_prefix(self):
        assert _bare_key("opponent_castled_status") == "castled_status"

    def test_no_prefix(self):
        assert _bare_key("material_difference") == "material_difference"


# ── influence ─────────────────────────────────────────────────────────────────

class TestScoreMoveInfluence:
    def _states(self) -> list[WeightedPrimitiveState]:
        return [
            _weighted_state("self_material_difference", "material", 0.9),
            _weighted_state("self_castled_status", "king_safety", 0.8),
            _weighted_state("self_bishop_activity", "piece_activity", 0.6),
        ]

    def test_good_move_sets_is_good(self):
        trace = _trace(cpl=10.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        assert record.is_good is True
        assert record.is_bad is False

    def test_bad_move_sets_is_bad(self):
        trace = _trace(cpl=150.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        assert record.is_bad is True
        assert record.is_good is False

    def test_top_n_respects_limit(self):
        trace = _trace(cpl=10.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev, top_n=2)
        assert len(record.top_primitives) == 2

    def test_top_primitives_sorted_by_score(self):
        trace = _trace(cpl=10.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        scores = [e.weighted_score for e in record.top_primitives]
        assert scores == sorted(scores, reverse=True)

    def test_bare_key_stripped(self):
        trace = _trace(cpl=10.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        bare_keys = {e.bare_key for e in record.top_primitives}
        assert "material_difference" in bare_keys
        assert not any(k.startswith("self_") for k in bare_keys)

    def test_no_cpl_neither_good_nor_bad(self):
        trace = _trace(cpl=None, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        assert record.is_good is False
        assert record.is_bad is False

    def test_middle_cpl_neither(self):
        trace = _trace(cpl=60.0, weighted_states=self._states())
        ev = evaluate_move(trace)
        record = score_move_influence(trace, ev)
        assert record.is_good is False
        assert record.is_bad is False


# ── rebalancer ────────────────────────────────────────────────────────────────

class TestRebalanceWeights:
    def _influence_good(self) -> InfluenceRecord:
        return InfluenceRecord(
            ply_index=0,
            quality_label="good",
            top_primitives=[
                PrimitiveInfluenceEntry(
                    primitive_id="self_material_difference",
                    bare_key="material_difference",
                    category="material",
                    weighted_score=0.9,
                )
            ],
            is_good=True,
            is_bad=False,
        )

    def _influence_bad(self) -> InfluenceRecord:
        return InfluenceRecord(
            ply_index=2,
            quality_label="blunder",
            top_primitives=[
                PrimitiveInfluenceEntry(
                    primitive_id="self_material_difference",
                    bare_key="material_difference",
                    category="material",
                    weighted_score=0.9,
                )
            ],
            is_good=False,
            is_bad=True,
        )

    def test_positive_influence_increases_weight(self):
        policy = _policy({"material_difference": 2.0})
        orig = {"material_difference": 2.0}
        adjustments = rebalance_weights([self._influence_good()], policy, orig, gain=0.1)
        adj = {a.bare_key: a for a in adjustments}
        assert adj["material_difference"].new_weight > 2.0

    def test_negative_influence_decreases_weight(self):
        policy = _policy({"material_difference": 2.0})
        orig = {"material_difference": 2.0}
        adjustments = rebalance_weights([self._influence_bad()], policy, orig, gain=0.1)
        adj = {a.bare_key: a for a in adjustments}
        assert adj["material_difference"].new_weight < 2.0

    def test_floor_clamp(self):
        policy = _policy({"material_difference": 0.2})
        orig = {"material_difference": 2.0}
        many_bad = [self._influence_bad() for _ in range(100)]
        adjustments = rebalance_weights(many_bad, policy, orig, gain=0.5, floor_factor=0.5)
        adj = {a.bare_key: a for a in adjustments}
        assert adj["material_difference"].new_weight >= orig["material_difference"] * 0.5
        assert adj["material_difference"].clamped is True

    def test_ceiling_clamp(self):
        policy = _policy({"material_difference": 2.0})
        orig = {"material_difference": 2.0}
        many_good = [self._influence_good() for _ in range(100)]
        adjustments = rebalance_weights(many_good, policy, orig, gain=0.5, ceiling_factor=2.0)
        adj = {a.bare_key: a for a in adjustments}
        assert adj["material_difference"].new_weight <= orig["material_difference"] * 2.0
        assert adj["material_difference"].clamped is True

    def test_no_influence_no_adjustments(self):
        empty_influence = [
            InfluenceRecord(ply_index=0, quality_label=None, top_primitives=[], is_good=False, is_bad=False)
        ]
        policy = _policy({"material_difference": 2.0})
        orig = {"material_difference": 2.0}
        adjustments = rebalance_weights(empty_influence, policy, orig)
        assert adjustments == []

    def test_unknown_key_skipped(self):
        influence = InfluenceRecord(
            ply_index=0,
            quality_label="good",
            top_primitives=[
                PrimitiveInfluenceEntry(
                    primitive_id="self_nonexistent",
                    bare_key="nonexistent",
                    category="misc",
                    weighted_score=0.8,
                )
            ],
            is_good=True,
            is_bad=False,
        )
        policy = _policy({"material_difference": 2.0})
        orig = {"material_difference": 2.0}
        adjustments = rebalance_weights([influence], policy, orig)
        keys = {a.bare_key for a in adjustments}
        assert "nonexistent" not in keys


# ── learning_log ──────────────────────────────────────────────────────────────

class TestLearningLog:
    def test_add_and_length(self):
        log = LearningLog(run_id="test_run")
        log.add(GameAdjustmentEntry(
            game_num=1, game_id="abc", ply_count=10,
            good_move_count=3, bad_move_count=2, adjustments=[],
        ))
        assert len(log.entries) == 1

    def test_save_creates_file(self, tmp_path):
        log = LearningLog(run_id="test_run")
        log.add(GameAdjustmentEntry(
            game_num=1, game_id="abc", ply_count=5,
            good_move_count=1, bad_move_count=0, adjustments=[],
        ))
        path = tmp_path / "learning_log.json"
        log.save(path)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["run_id"] == "test_run"
        assert len(data["entries"]) == 1


# ── summary ───────────────────────────────────────────────────────────────────

class TestBuildGameSummary:
    def test_avg_cpl_computed(self):
        traces = [_trace(cpl=100.0), _trace(cpl=200.0)]
        row = build_game_summary("run1", 1, "game1", traces, [])
        assert row.avg_cpl == pytest.approx(150.0, rel=0.01)

    def test_no_cpl_returns_none(self):
        traces = [_trace(cpl=None)]
        row = build_game_summary("run1", 1, "game1", traces, [])
        assert row.avg_cpl is None

    def test_blunder_counted(self):
        traces = [_trace(cpl=300.0, blunder_label="blunder"), _trace(cpl=10.0)]
        row = build_game_summary("run1", 1, "game1", traces, [])
        assert row.blunder_count == 1

    def test_rank_1_rate(self):
        traces = [
            _trace(cpl=10.0, selected_rank=1),
            _trace(cpl=80.0, selected_rank=2),
        ]
        row = build_game_summary("run1", 1, "game1", traces, [])
        assert row.rank_1_rate == pytest.approx(0.5)


class TestBuildPrimitiveSummaries:
    def test_total_delta_computed(self):
        initial = {"material_difference": 2.0}
        final = {"material_difference": 2.2}
        adjustments = [
            WeightAdjustment(
                bare_key="material_difference", old_weight=2.0, new_weight=2.2,
                delta=0.2, net_influence=4.0, clamped=False,
            )
        ]
        summaries = build_primitive_summaries(initial, final, adjustments)
        assert len(summaries) == 1
        assert summaries[0].total_delta == pytest.approx(0.2, rel=0.01)

    def test_key_in_initial_not_final_excluded(self):
        initial = {"material_difference": 2.0, "removed_key": 1.0}
        final = {"material_difference": 2.1}
        summaries = build_primitive_summaries(initial, final, [])
        keys = {s.bare_key for s in summaries}
        assert "removed_key" not in keys


# ── learning_runner passthrough (no learning) ─────────────────────────────────

class TestLearningRunnerPassthrough:
    def test_learning_mode_false_config(self):
        """Verify learning_baseline config can be loaded and learning_mode key parsed."""
        import yaml
        with open("configs/experiments/learning_baseline.yaml") as f:
            config = yaml.safe_load(f)
        # Confirm the config has learning parameters
        assert config.get("learning_mode") is True
        assert config.get("gain") == 0.05
        assert config.get("good_threshold") == 30.0
        assert config.get("bad_threshold") == 100.0

    def test_save_learned_policy_creates_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        policy = _policy({"material_difference": 2.1})
        path = save_learned_policy(policy, "balanced", "test_run_20260101_000000")
        assert path.exists()
        import yaml
        data = yaml.safe_load(path.read_text())
        assert data["weight_map"]["material_difference"] == pytest.approx(2.1)

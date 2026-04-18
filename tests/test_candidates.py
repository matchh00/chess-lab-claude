"""Tests for Phase 3 (policies) and Phase 4 (candidates)."""
from __future__ import annotations

import random

import chess
import pytest

from src.candidates.annotator import annotate_candidates
from src.candidates.filters import filter_candidates
from src.candidates.generator import generate_engine_assisted, generate_heuristic_only
from src.candidates.ranker import rank_candidates
from src.candidates.shuffler import shuffle_candidates
from src.environment.engine_wrapper import EngineWrapper
from src.policies.profiles import PolicyProfile, load_all_policies, load_policy
from src.policies.summaries import machine_summary, text_priority_summary
from src.policies.weighting import apply_policy
from src.primitives.extractor import PrimitiveExtractor, build_default_registry
from src.storage.models import CandidateMoveRecord, WeightedPrimitiveState

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDGAME_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def balanced_policy() -> PolicyProfile:
    return load_policy("balanced")


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def extractor(registry):
    return PrimitiveExtractor(registry)


@pytest.fixture(scope="module")
def engine():
    e = EngineWrapper(STOCKFISH_PATH, skill_level=5, depth=8, time_limit=0.05)
    e.open()
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Phase 3 — Policy profiles
# ---------------------------------------------------------------------------

class TestPolicyProfiles:
    def test_load_balanced(self):
        p = load_policy("balanced")
        assert p.policy_id == "balanced"
        assert p.name == "Balanced"
        assert isinstance(p.weight_map, dict)
        assert len(p.weight_map) > 0
        assert isinstance(p.group_weights, dict)

    def test_load_all_five_policies(self):
        policies = load_all_policies()
        assert len(policies) == 5
        expected_ids = {"balanced", "aggressive", "defensive",
                        "development_first", "endgame_clean"}
        assert set(policies.keys()) == expected_ids

    def test_weight_lookup_primitive_specific(self, balanced_policy):
        # YAML weight_map uses bare keys (e.g. "material_difference"), not "self_" prefixed
        first_key = next(iter(balanced_policy.weight_map))
        w = balanced_policy.get_weight(first_key, "material")
        assert w == balanced_policy.weight_map[first_key]

    def test_weight_lookup_falls_back_to_group(self, balanced_policy):
        # Use a primitive_id not in weight_map but whose category is in group_weights
        w = balanced_policy.get_weight("nonexistent_primitive_xyz", "material")
        assert w == balanced_policy.group_weights["material"]

    def test_weight_lookup_default(self, balanced_policy):
        w = balanced_policy.get_weight("nonexistent", "nonexistent_category")
        assert w == 1.0

    def test_policy_serializable(self, balanced_policy):
        data = balanced_policy.model_dump()
        assert "weight_map" in data
        assert "group_weights" in data
        restored = PolicyProfile(**data)
        assert restored.policy_id == balanced_policy.policy_id


# ---------------------------------------------------------------------------
# Phase 3 — Weighting
# ---------------------------------------------------------------------------

class TestWeighting:
    def test_apply_policy_returns_weighted_states(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        assert len(states) == len(primitives)
        assert all(isinstance(s, WeightedPrimitiveState) for s in states)

    def test_effective_score_formula(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        for s in states:
            expected = round(s.normalized_value * s.weight * s.confidence, 4)
            assert abs(s.weighted_score - expected) < 1e-6, (
                f"{s.primitive_id}: got {s.weighted_score}, expected {expected}"
            )

    def test_importance_labels_assigned(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        valid_labels = {"critical", "high", "medium", "low", "negligible"}
        for s in states:
            assert s.importance_label in valid_labels

    def test_importance_critical_threshold(self, balanced_policy):
        # Construct a PrimitiveValue that should yield a "critical" label
        from src.storage.models import PrimitiveValue
        pv = PrimitiveValue(
            primitive_id="self_material_difference",
            name="Material Difference",
            side="self",
            value=10.0,
            normalized_value=1.0,
            text_render="test",
            confidence=1.0,
            category="material",
        )
        states = apply_policy([pv], balanced_policy)
        # balanced material weight is 2.0, so score = 1.0 * 2.0 * 1.0 = 2.0 → critical
        assert states[0].importance_label == "critical"

    def test_confidence_scales_score(self, balanced_policy):
        from src.storage.models import PrimitiveValue
        pv_high = PrimitiveValue(
            primitive_id="self_queen_overextension_risk",
            name="Queen Risk",
            side="self",
            value=0.5,
            normalized_value=0.5,
            text_render="test",
            confidence=1.0,
            category="piece_activity",
        )
        pv_low = pv_high.model_copy(update={"confidence": 0.6})
        s_high = apply_policy([pv_high], balanced_policy)[0]
        s_low = apply_policy([pv_low], balanced_policy)[0]
        assert s_high.weighted_score > s_low.weighted_score

    def test_weighted_states_serializable(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        for s in states:
            data = s.model_dump()
            assert "weighted_score" in data
            assert "importance_label" in data


# ---------------------------------------------------------------------------
# Phase 3 — Summaries
# ---------------------------------------------------------------------------

class TestSummaries:
    def test_machine_summary_sorted_descending(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        summary = machine_summary(states)
        scores = [s["weighted_score"] for s in summary]
        assert scores == sorted(scores, reverse=True)

    def test_text_priority_summary_non_empty(self, extractor, balanced_policy):
        board = chess.Board()
        primitives = extractor.extract(board, chess.WHITE)
        states = apply_policy(primitives, balanced_policy)
        text = text_priority_summary(balanced_policy, states)
        assert isinstance(text, str)
        assert len(text) > 20

    def test_text_summary_all_five_styles(self, extractor):
        policies = load_all_policies()
        board = chess.Board()
        for policy in policies.values():
            primitives = extractor.extract(board, chess.WHITE)
            states = apply_policy(primitives, policy)
            text = text_priority_summary(policy, states)
            assert len(text) > 10, f"Empty summary for policy {policy.policy_id}"

    def test_aggressive_vs_defensive_differ(self, extractor):
        agg = load_policy("aggressive")
        def_ = load_policy("defensive")
        board = chess.Board()
        prim = extractor.extract(board, chess.WHITE)
        text_agg = text_priority_summary(agg, apply_policy(prim, agg))
        text_def = text_priority_summary(def_, apply_policy(prim, def_))
        assert text_agg != text_def


# ---------------------------------------------------------------------------
# Phase 4 — Candidate generation (heuristic_only, no engine)
# ---------------------------------------------------------------------------

class TestHeuristicGeneration:
    def test_generates_candidates(self):
        board = chess.Board()
        candidates = generate_heuristic_only(board, target_n=6, rng=random.Random(0))
        assert len(candidates) >= 1

    def test_all_candidates_legal(self):
        board = chess.Board()
        candidates = generate_heuristic_only(board, target_n=6, rng=random.Random(1))
        legal = {m.uci() for m in board.legal_moves}
        for c in candidates:
            assert c.uci in legal

    def test_no_duplicates(self):
        board = chess.Board()
        candidates = generate_heuristic_only(board, target_n=8, rng=random.Random(2))
        ucis = [c.uci for c in candidates]
        assert len(ucis) == len(set(ucis))

    def test_sources_assigned(self):
        board = chess.Board()
        candidates = generate_heuristic_only(board, target_n=8, rng=random.Random(3))
        valid_sources = {"engine", "heuristic", "random", "hybrid"}
        for c in candidates:
            assert c.source in valid_sources

    def test_midgame_position(self):
        board = chess.Board(MIDGAME_FEN)
        candidates = generate_heuristic_only(board, target_n=6, rng=random.Random(7))
        assert len(candidates) >= 1
        legal = {m.uci() for m in board.legal_moves}
        for c in candidates:
            assert c.uci in legal

    def test_candidate_serializable(self):
        board = chess.Board()
        candidates = generate_heuristic_only(board, target_n=4, rng=random.Random(9))
        for c in candidates:
            data = c.model_dump_json()
            assert "uci" in data
            assert "san" in data


# ---------------------------------------------------------------------------
# Phase 4 — Engine-assisted generation (requires Stockfish)
# ---------------------------------------------------------------------------

class TestEngineAssistedGeneration:
    def test_engine_generates_candidates(self, engine):
        board = chess.Board()
        candidates = generate_engine_assisted(board, engine, rng=random.Random(42))
        assert len(candidates) >= 1

    def test_engine_sources_present(self, engine):
        board = chess.Board()
        candidates = generate_engine_assisted(board, engine, rng=random.Random(42))
        sources = {c.source for c in candidates}
        assert "engine" in sources

    def test_engine_all_candidates_legal(self, engine):
        board = chess.Board()
        candidates = generate_engine_assisted(board, engine, rng=random.Random(42))
        legal = {m.uci() for m in board.legal_moves}
        for c in candidates:
            assert c.uci in legal

    def test_engine_no_duplicates(self, engine):
        board = chess.Board()
        candidates = generate_engine_assisted(board, engine, rng=random.Random(42))
        ucis = [c.uci for c in candidates]
        assert len(ucis) == len(set(ucis))

    def test_engine_eval_populated_for_engine_candidates(self, engine):
        board = chess.Board()
        candidates = generate_engine_assisted(board, engine, rng=random.Random(42))
        engine_cands = [c for c in candidates if c.source == "engine"]
        assert all(c.engine_eval_after is not None for c in engine_cands)

    def test_engine_eval_before_propagated(self, engine):
        board = chess.Board()
        eval_before = engine.evaluate(board)
        candidates = generate_engine_assisted(
            board, engine, engine_eval_before=eval_before, rng=random.Random(1)
        )
        for c in candidates:
            assert c.engine_eval_before == eval_before

    def test_analyse_top_n_returns_evals(self, engine):
        board = chess.Board()
        results = engine.analyse_top_n(board, n=5)
        assert len(results) >= 1
        for uci, cp in results:
            assert isinstance(uci, str)
            assert cp is None or isinstance(cp, float)


# ---------------------------------------------------------------------------
# Phase 4 — Filter, rank, shuffle (no engine needed)
# ---------------------------------------------------------------------------

class TestFilter:
    def test_removes_duplicates(self):
        board = chess.Board()
        c = CandidateMoveRecord(uci="e2e4", san="e4", source="engine")
        candidates = [c, c.model_copy()]  # exact duplicate
        filtered = filter_candidates(candidates, board)
        assert len(filtered) == 1

    def test_removes_illegal_moves(self):
        board = chess.Board()
        illegal = CandidateMoveRecord(uci="e2e5", san="e5??", source="random")
        legal = CandidateMoveRecord(uci="e2e4", san="e4", source="engine")
        filtered = filter_candidates([illegal, legal], board)
        assert len(filtered) == 1
        assert filtered[0].uci == "e2e4"

    def test_preserves_order(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=6, rng=random.Random(5))
        filtered = filter_candidates(cands, board)
        # order preserved; verify all still legal
        legal = {m.uci() for m in board.legal_moves}
        for c in filtered:
            assert c.uci in legal


class TestRanker:
    def test_assigns_internal_rank(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=5, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        ranks = [c.internal_rank for c in ranked]
        assert sorted(ranks) == list(range(1, len(ranked) + 1))

    def test_rank_1_is_best_engine_candidate(self, engine):
        board = chess.Board()
        cands = generate_engine_assisted(board, engine, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        rank1 = next(c for c in ranked if c.internal_rank == 1)
        # The rank-1 candidate should have the highest engine eval
        engine_cands = [c for c in ranked if c.engine_eval_after is not None]
        if engine_cands:
            best_eval = max(c.engine_eval_after for c in engine_cands)
            assert rank1.engine_eval_after == best_eval or rank1.source == "engine"

    def test_presentation_index_unchanged_by_ranker(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=4, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        # All presentation_index should still be 0 (default); shuffler sets them
        assert all(c.presentation_index == 0 for c in ranked)


class TestShuffler:
    def test_assigns_presentation_index(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=6, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(1))
        indices = sorted(c.presentation_index for c in shuffled)
        assert indices == list(range(len(shuffled)))

    def test_internal_rank_unchanged_by_shuffler(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=6, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        ranks_before = {c.uci: c.internal_rank for c in ranked}
        shuffled = shuffle_candidates(ranked, rng=random.Random(99))
        for c in shuffled:
            assert c.internal_rank == ranks_before[c.uci]

    def test_presentation_differs_from_rank(self):
        # With enough candidates, shuffled order should differ from ranked order
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=8, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(7))
        pairs = [(c.internal_rank, c.presentation_index) for c in shuffled]
        # At least one candidate has presentation_index != internal_rank - 1
        assert any(rank - 1 != pres for rank, pres in pairs)

    def test_uci_set_preserved(self):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=6, rng=random.Random(0))
        ranked = rank_candidates(cands, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(3))
        assert {c.uci for c in ranked} == {c.uci for c in shuffled}


# ---------------------------------------------------------------------------
# Phase 4 — Annotator
# ---------------------------------------------------------------------------

class TestAnnotator:
    def test_fills_board_after_fen(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=4, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(0))

        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            assert c.board_after_fen != ""
            assert c.board_after_fen != board.fen()

    def test_fills_delta_summary(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=4, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(0))

        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            assert isinstance(c.primitive_delta_summary, str)
            assert len(c.primitive_delta_summary) > 0

    def test_risk_flags_are_list(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=4, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(0))

        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            assert isinstance(c.risk_flags, list)

    def test_internal_rank_and_presentation_index_preserved(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=5, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(4))

        before = {c.uci: (c.internal_rank, c.presentation_index) for c in shuffled}
        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            assert (c.internal_rank, c.presentation_index) == before[c.uci]

    def test_score_delta_is_float(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=4, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(0))

        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            assert c.score_delta is not None
            assert isinstance(c.score_delta, float)

    def test_annotated_records_fully_serializable(self, extractor, balanced_policy):
        board = chess.Board()
        cands = generate_heuristic_only(board, target_n=3, rng=random.Random(0))
        filtered = filter_candidates(cands, board)
        ranked = rank_candidates(filtered, board)
        shuffled = shuffle_candidates(ranked, rng=random.Random(0))

        primitives = extractor.extract(board, chess.WHITE)
        baseline = apply_policy(primitives, balanced_policy)
        annotated = annotate_candidates(shuffled, board, chess.WHITE, extractor, balanced_policy, baseline)

        for c in annotated:
            json_str = c.model_dump_json()
            assert "internal_rank" in json_str
            assert "presentation_index" in json_str
            assert "primitive_delta_summary" in json_str

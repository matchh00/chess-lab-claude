from __future__ import annotations

import json

import chess
import pytest

from src.llm.schemas import LLMDecisionResponse, parse_response
from src.llm.decision import choose_move, _fallback_decision
from src.llm.prompt_version import CURRENT_VERSION, get_template_path, get_version
from src.llm.prompts import build_prompt
from src.narratives.token_budget import count_tokens
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionPromptRecord,
    DecisionRecord,
    MoveTrace,
)


# ── LLMDecisionResponse / parse_response ─────────────────────────────────────

class TestParseResponse:
    def _valid_json(self, move="e2e4", rank=1, reason="Good move.", confidence=0.9):
        return json.dumps({
            "selected_move": move,
            "candidate_rank": rank,
            "reason": reason,
            "confidence": confidence,
        })

    def test_bare_json(self):
        raw = self._valid_json()
        result = parse_response(raw)
        assert isinstance(result, LLMDecisionResponse)
        assert result.selected_move == "e2e4"

    def test_markdown_fenced(self):
        raw = f"```json\n{self._valid_json()}\n```"
        result = parse_response(raw)
        assert result.selected_move == "e2e4"

    def test_markdown_fenced_no_lang(self):
        raw = f"```\n{self._valid_json()}\n```"
        result = parse_response(raw)
        assert result.selected_move == "e2e4"

    def test_json_embedded_in_text(self):
        raw = f"Here is my answer:\n{self._valid_json()}\nThat's it."
        result = parse_response(raw)
        assert result.selected_move == "e2e4"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_response("not json at all")

    def test_confidence_bounds(self):
        raw = self._valid_json(confidence=0.5)
        result = parse_response(raw)
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_out_of_range_raises(self):
        raw = json.dumps({
            "selected_move": "e2e4",
            "candidate_rank": 1,
            "reason": "Good.",
            "confidence": 1.5,
        })
        with pytest.raises(Exception):
            parse_response(raw)

    def test_missing_field_raises(self):
        raw = json.dumps({"selected_move": "e2e4", "candidate_rank": 1})
        with pytest.raises(Exception):
            parse_response(raw)


# ── prompt_version ────────────────────────────────────────────────────────────

class TestPromptVersion:
    def test_current_version_is_string(self):
        assert isinstance(CURRENT_VERSION, str)
        assert CURRENT_VERSION.startswith("v")

    def test_get_version_returns_current(self):
        assert get_version() == CURRENT_VERSION

    def test_get_template_path_returns_string(self):
        path = get_template_path()
        assert isinstance(path, str)
        assert path.endswith(".md")

    def test_get_template_path_file_exists(self):
        import os
        path = get_template_path()
        assert os.path.exists(path)

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError):
            get_template_path("v99.99")


# ── build_prompt ──────────────────────────────────────────────────────────────

def _make_snapshot() -> BoardSnapshot:
    return BoardSnapshot(
        game_id="test-game",
        ply_index=0,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        turn="white",
        legal_moves=["e2e4", "d2d4", "g1f3"],
        move_history=[],
        is_check=False,
        is_game_over=False,
    )


def _make_candidate(uci: str, san: str, rank: int = 1, pidx: int = 0) -> CandidateMoveRecord:
    return CandidateMoveRecord(
        uci=uci,
        san=san,
        internal_rank=rank,
        presentation_index=pidx,
        source="engine",
        engine_eval_before=0.0,
        engine_eval_after=20.0,
        risk_flags=[],
        primitives_after=[],
    )


class TestBuildPrompt:
    def test_returns_decision_prompt_record(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4", rank=1, pidx=0)]
        result = build_prompt(snap, "Good position.", "Balanced policy.", candidates)
        assert isinstance(result, DecisionPromptRecord)

    def test_system_prompt_not_empty(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4")]
        result = build_prompt(snap, "", "", candidates)
        assert len(result.system_prompt) > 0

    def test_user_prompt_contains_fen(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4")]
        result = build_prompt(snap, "Narrative.", "Policy.", candidates)
        assert snap.fen in result.user_prompt

    def test_candidate_block_contains_san(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4", rank=1, pidx=0)]
        result = build_prompt(snap, "", "", candidates)
        assert "e4" in result.candidate_block

    def test_candidate_block_contains_uci(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4")]
        result = build_prompt(snap, "", "", candidates)
        assert "e2e4" in result.candidate_block

    def test_token_count_positive(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4")]
        result = build_prompt(snap, "Narrative.", "Policy.", candidates)
        assert result.token_count > 0

    def test_version_stored(self):
        snap = _make_snapshot()
        candidates = [_make_candidate("e2e4", "e4")]
        result = build_prompt(snap, "", "", candidates)
        assert result.prompt_version == CURRENT_VERSION

    def test_candidates_ordered_by_presentation_index(self):
        snap = _make_snapshot()
        candidates = [
            _make_candidate("d2d4", "d4", rank=2, pidx=0),
            _make_candidate("e2e4", "e4", rank=1, pidx=1),
        ]
        result = build_prompt(snap, "", "", candidates)
        d4_pos = result.candidate_block.index("d4")
        e4_pos = result.candidate_block.index("e4")
        assert d4_pos < e4_pos


# ── choose_move / decision ────────────────────────────────────────────────────

def _make_prompt_record() -> DecisionPromptRecord:
    return DecisionPromptRecord(
        system_prompt="You are a chess engine.",
        user_prompt="Pick a move.",
        prompt_version="v1.0",
    )


def _make_candidates_for_board(board: chess.Board) -> list[CandidateMoveRecord]:
    moves = list(board.legal_moves)[:3]
    candidates = []
    for i, move in enumerate(moves):
        board_copy = board.copy()
        board_copy.push(move)
        san = board.san(move)
        candidates.append(CandidateMoveRecord(
            uci=move.uci(),
            san=san,
            internal_rank=i + 1,
            presentation_index=i,
            source="engine",
            engine_eval_before=0.0,
            engine_eval_after=0.0,
            risk_flags=[],
            primitives_after=[],
        ))
    return candidates


class TestChooseMove:
    def _board(self) -> chess.Board:
        return chess.Board()

    def test_valid_llm_response(self):
        board = self._board()
        candidates = _make_candidates_for_board(board)
        first_uci = candidates[0].uci

        def mock_llm(sys, usr):
            return json.dumps({
                "selected_move": first_uci,
                "candidate_rank": 1,
                "reason": "Best move.",
                "confidence": 0.8,
            })

        record = choose_move(_make_prompt_record(), candidates, board, llm_call_fn=mock_llm)
        assert isinstance(record, DecisionRecord)
        assert record.selected_uci == first_uci
        assert record.fallback_used is False
        assert record.is_valid is True

    def test_invalid_json_triggers_retry_then_fallback(self):
        board = self._board()
        candidates = _make_candidates_for_board(board)
        call_count = [0]

        def mock_llm(sys, usr):
            call_count[0] += 1
            return "not json"

        record = choose_move(_make_prompt_record(), candidates, board, llm_call_fn=mock_llm)
        assert call_count[0] == 2  # first attempt + retry
        assert record.fallback_used is True
        assert record.is_valid is False

    def test_invalid_move_triggers_fallback(self):
        board = self._board()
        candidates = _make_candidates_for_board(board)

        def mock_llm(sys, usr):
            return json.dumps({
                "selected_move": "z9z9",  # not a real move
                "candidate_rank": 1,
                "reason": "Bad.",
                "confidence": 0.1,
            })

        record = choose_move(_make_prompt_record(), candidates, board, llm_call_fn=mock_llm)
        assert record.fallback_used is True

    def test_retry_succeeds(self):
        board = self._board()
        candidates = _make_candidates_for_board(board)
        first_uci = candidates[0].uci
        call_count = [0]

        def mock_llm(sys, usr):
            call_count[0] += 1
            if call_count[0] == 1:
                return "garbage"
            return json.dumps({
                "selected_move": first_uci,
                "candidate_rank": 1,
                "reason": "Good.",
                "confidence": 0.9,
            })

        record = choose_move(_make_prompt_record(), candidates, board, llm_call_fn=mock_llm)
        assert record.selected_uci == first_uci
        assert record.fallback_used is False
        assert call_count[0] == 2

    def test_fallback_selects_rank_1(self):
        board = self._board()
        candidates = _make_candidates_for_board(board)
        best = min(candidates, key=lambda c: c.internal_rank)
        record = _fallback_decision(candidates)
        assert record.selected_uci == best.uci
        assert record.fallback_used is True


# ── storage models (DecisionPromptRecord, DecisionRecord, MoveTrace) ──────────

class TestDecisionPromptRecord:
    def test_defaults(self):
        rec = DecisionPromptRecord(system_prompt="sys", user_prompt="usr")
        assert rec.model == "claude-sonnet-4-6"
        assert rec.temperature == 0.2
        assert rec.prompt_version == "v1.1"
        assert rec.token_count == 0

    def test_to_json(self):
        rec = DecisionPromptRecord(system_prompt="sys", user_prompt="usr")
        data = json.loads(rec.to_json())
        assert data["system_prompt"] == "sys"


class TestDecisionRecord:
    def test_valid_record(self):
        rec = DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=1,
            selected_presentation_index=0,
        )
        assert rec.is_valid is True
        assert rec.fallback_used is False
        assert rec.confidence == 0.0

    def test_to_json(self):
        rec = DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=1,
            selected_presentation_index=0,
        )
        data = json.loads(rec.to_json())
        assert data["selected_uci"] == "e2e4"


class TestMoveTrace:
    def test_minimal_construction(self):
        snap = _make_snapshot()
        trace = MoveTrace(
            game_id="test",
            ply_index=0,
            board_snapshot=snap,
        )
        assert trace.game_id == "test"
        assert trace.blunder_label is None
        assert trace.candidates == []

    def test_to_json(self):
        snap = _make_snapshot()
        trace = MoveTrace(game_id="g1", ply_index=0, board_snapshot=snap)
        data = json.loads(trace.to_json())
        assert data["game_id"] == "g1"

    def test_with_decision_record(self):
        snap = _make_snapshot()
        dec = DecisionRecord(
            selected_uci="e2e4",
            selected_internal_rank=1,
            selected_presentation_index=0,
        )
        trace = MoveTrace(
            game_id="g1",
            ply_index=1,
            board_snapshot=snap,
            decision_record=dec,
        )
        assert trace.decision_record is not None
        assert trace.decision_record.selected_uci == "e2e4"

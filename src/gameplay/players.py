from __future__ import annotations

import logging
from typing import Callable, Optional

import chess

from src.llm.schemas import parse_response
from src.storage.models import DecisionRecord

logger = logging.getLogger(__name__)

_LLM_RAW_SYSTEM = (
    "You are a chess engine. Given a FEN position and a list of legal moves (in UCI format), "
    "select the best move. Respond ONLY with a valid JSON object.\n\n"
    "Response format:\n"
    '{"selected_move": "<uci>", "candidate_rank": 1, "reason": "<brief reason>", "confidence": <0.0-1.0>}'
)


class LLMRawPlayer:
    """Baseline player that sends only FEN + legal moves to the LLM, no primitives or narrative."""

    def __init__(self, llm_call_fn: Optional[Callable[[str, str], str]] = None) -> None:
        if llm_call_fn is None:
            from src.llm.client import call_llm
            llm_call_fn = call_llm
        self._llm_call_fn = llm_call_fn

    def decide(self, board: chess.Board) -> DecisionRecord:
        legal_ucis = [m.uci() for m in board.legal_moves]
        user_prompt = (
            f"FEN: {board.fen()}\n"
            f"Legal moves (UCI): {', '.join(legal_ucis)}\n\n"
            "Select the best move. Respond with JSON only."
        )

        raw = self._llm_call_fn(_LLM_RAW_SYSTEM, user_prompt)
        try:
            parsed = parse_response(raw)
        except ValueError as e:
            logger.warning("LLMRawPlayer parse error: %s — using first legal move.", e)
            return DecisionRecord(
                selected_uci=legal_ucis[0],
                selected_internal_rank=1,
                selected_presentation_index=0,
                reasoning_summary="Fallback: parse failed.",
                confidence=0.0,
                fallback_used=True,
                is_valid=False,
            )

        if parsed.selected_move not in set(legal_ucis):
            logger.warning("LLMRawPlayer selected illegal move %r — using first legal.", parsed.selected_move)
            return DecisionRecord(
                selected_uci=legal_ucis[0],
                selected_internal_rank=1,
                selected_presentation_index=0,
                reasoning_summary="Fallback: illegal move selected.",
                confidence=0.0,
                fallback_used=True,
                is_valid=False,
            )

        return DecisionRecord(
            selected_uci=parsed.selected_move,
            selected_internal_rank=parsed.candidate_rank,
            selected_presentation_index=0,
            reasoning_summary=parsed.reason,
            confidence=parsed.confidence,
            fallback_used=False,
            is_valid=True,
        )

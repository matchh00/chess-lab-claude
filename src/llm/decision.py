from __future__ import annotations

import logging
from typing import Callable, Optional

import chess

from src.llm.schemas import parse_response
from src.storage.models import CandidateMoveRecord, DecisionPromptRecord, DecisionRecord

logger = logging.getLogger(__name__)

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed or the selected move was invalid. "
    "Respond ONLY with a JSON object matching the schema. "
    "selected_move MUST be one of the UCI strings listed in the candidate moves."
)


def _fallback_decision(candidates: list[CandidateMoveRecord]) -> DecisionRecord:
    best = min(candidates, key=lambda c: c.internal_rank)
    return DecisionRecord(
        selected_uci=best.uci,
        selected_internal_rank=best.internal_rank,
        selected_presentation_index=best.presentation_index,
        reasoning_summary="Fallback: selected highest-ranked candidate after LLM parse failure.",
        confidence=0.0,
        fallback_used=True,
        is_valid=False,
    )


def _validate_and_build(
    raw_text: str,
    candidates: list[CandidateMoveRecord],
    board: chess.Board,
) -> Optional[DecisionRecord]:
    try:
        parsed = parse_response(raw_text)
    except ValueError as e:
        logger.warning("LLM parse error: %s", e)
        return None

    legal_ucis = {m.uci() for m in board.legal_moves}
    candidate_ucis = {c.uci for c in candidates}

    if parsed.selected_move not in candidate_ucis:
        logger.warning("LLM selected move %r not in candidate list.", parsed.selected_move)
        return None
    if parsed.selected_move not in legal_ucis:
        logger.warning("LLM selected move %r is illegal.", parsed.selected_move)
        return None

    matching = [c for c in candidates if c.uci == parsed.selected_move]
    if not matching:
        return None
    chosen = matching[0]

    return DecisionRecord(
        selected_uci=parsed.selected_move,
        selected_internal_rank=chosen.internal_rank,
        selected_presentation_index=chosen.presentation_index,
        reasoning_summary=parsed.reason,
        confidence=parsed.confidence,
        fallback_used=False,
        is_valid=True,
    )


def choose_move(
    prompt_record: DecisionPromptRecord,
    candidates: list[CandidateMoveRecord],
    board: chess.Board,
    llm_call_fn: Optional[Callable[[str, str], str]] = None,
) -> DecisionRecord:
    """Call LLM, parse response, validate move. Falls back to internal_rank=1 on failure."""
    if llm_call_fn is None:
        from src.llm.client import call_llm
        llm_call_fn = call_llm

    # First attempt
    raw = llm_call_fn(prompt_record.system_prompt, prompt_record.user_prompt)
    result = _validate_and_build(raw, candidates, board)
    if result is not None:
        return result

    # Retry with stricter prompt
    logger.info("LLM first attempt failed — retrying with stricter prompt.")
    retry_prompt = prompt_record.user_prompt + _RETRY_SUFFIX
    raw2 = llm_call_fn(prompt_record.system_prompt, retry_prompt)
    result2 = _validate_and_build(raw2, candidates, board)
    if result2 is not None:
        return result2

    logger.warning("LLM retry also failed — using fallback.")
    return _fallback_decision(candidates)

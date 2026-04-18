from __future__ import annotations

import chess

from src.storage.models import CandidateMoveRecord

# Source priority bonuses in centipawn scale (used when engine eval is absent)
_SOURCE_BONUS: dict[str, float] = {
    "engine":    50.0,
    "heuristic": 20.0,
    "hybrid":    30.0,
    "random":     0.0,
}


def _score(candidate: CandidateMoveRecord, board: chess.Board) -> float:
    """Internal scoring function. Engine eval is authoritative when present."""
    if candidate.engine_eval_after is not None:
        # Engine eval already encodes move quality comprehensively.
        # Add a tiny source bonus so engine-sourced ties break in favour of engine.
        return candidate.engine_eval_after + _SOURCE_BONUS.get(candidate.source, 0.0) * 0.01
    # Fallback heuristics when no engine eval is available
    score = _SOURCE_BONUS.get(candidate.source, 0.0)
    move = chess.Move.from_uci(candidate.uci)
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            victim_values = {chess.PAWN: 10, chess.KNIGHT: 30, chess.BISHOP: 30,
                             chess.ROOK: 50, chess.QUEEN: 90, chess.KING: 0}
            score += victim_values.get(captured.piece_type, 0)
    if board.gives_check(move):
        score += 20.0
    if move.promotion:
        score += 90.0
    return score


def rank_candidates(
    candidates: list[CandidateMoveRecord],
    board: chess.Board,
) -> list[CandidateMoveRecord]:
    """Assign internal_rank (1-based, 1 = best) based on internal scoring.

    Returns a new list sorted by rank with internal_rank set.
    presentation_index is left unchanged (set later by shuffler).
    """
    scored = [(_score(c, board), i, c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)  # descending: higher score = better rank

    ranked: list[CandidateMoveRecord] = []
    for rank, (_, _, c) in enumerate(scored, start=1):
        ranked.append(c.model_copy(update={"internal_rank": rank}))
    return ranked

from __future__ import annotations

import random
from typing import Optional

import chess

from src.environment.engine_wrapper import EngineWrapper
from src.storage.models import CandidateMoveRecord

# Heuristic piece values for sorting forcing moves by capture value
_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

_ENGINE_TOP_N = 5
_MAX_FORCING = 2
_MAX_EXPLORATORY = 1


def _san(board: chess.Board, uci: str) -> str:
    return board.san(chess.Move.from_uci(uci))


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    captured = board.piece_at(move.to_square)
    if captured:
        return _PIECE_VALUES.get(captured.piece_type, 0)
    if board.is_en_passant(move):
        return 1
    return 0


def generate_engine_assisted(
    board: chess.Board,
    engine: EngineWrapper,
    engine_eval_before: Optional[float] = None,
    top_n: int = _ENGINE_TOP_N,
    max_forcing: int = _MAX_FORCING,
    max_exploratory: int = _MAX_EXPLORATORY,
    rng: Optional[random.Random] = None,
) -> list[CandidateMoveRecord]:
    """Generate candidates: engine top-N + forcing supplements + 1 exploratory."""
    if rng is None:
        rng = random.Random()

    candidates: list[CandidateMoveRecord] = []
    seen_ucis: set[str] = set()

    # 1. Engine top-N with evals
    engine_moves = engine.analyse_top_n(board, n=top_n)
    for uci, cp_after in engine_moves:
        if uci in seen_ucis:
            continue
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        candidates.append(CandidateMoveRecord(
            uci=uci,
            san=_san(board, uci),
            source="engine",
            engine_eval_before=engine_eval_before,
            engine_eval_after=cp_after,
        ))
        seen_ucis.add(uci)

    # 2. Forcing moves not already covered (captures + checks), sorted by capture value
    forcing: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        uci = move.uci()
        if uci in seen_ucis:
            continue
        is_forcing = board.is_capture(move) or board.gives_check(move)
        if is_forcing:
            forcing.append((_capture_value(board, move), move))

    forcing.sort(key=lambda x: x[0], reverse=True)
    for _, move in forcing[:max_forcing]:
        uci = move.uci()
        candidates.append(CandidateMoveRecord(
            uci=uci,
            san=_san(board, uci),
            source="heuristic",
            engine_eval_before=engine_eval_before,
        ))
        seen_ucis.add(uci)

    # 3. One random exploratory move not already in candidate set
    if max_exploratory > 0:
        legal_ucis = [m.uci() for m in board.legal_moves if m.uci() not in seen_ucis]
        if legal_ucis:
            exp_uci = rng.choice(legal_ucis)
            candidates.append(CandidateMoveRecord(
                uci=exp_uci,
                san=_san(board, exp_uci),
                source="random",
                engine_eval_before=engine_eval_before,
                notes="exploratory",
            ))

    return candidates


def generate_heuristic_only(
    board: chess.Board,
    target_n: int = 8,
    rng: Optional[random.Random] = None,
) -> list[CandidateMoveRecord]:
    """Generate candidates using heuristics only (no engine). Phase 4 stub."""
    if rng is None:
        rng = random.Random()

    candidates: list[CandidateMoveRecord] = []
    seen_ucis: set[str] = set()

    # Priority 1: captures sorted by most-valuable-victim least-valuable-attacker
    captures: list[tuple[float, chess.Move]] = []
    for move in board.legal_moves:
        if board.is_capture(move):
            victim_val = _capture_value(board, move)
            attacker = board.piece_at(move.from_square)
            attacker_val = _PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
            # MVV-LVA: high victim value, low attacker value → high score
            score = victim_val * 10 - attacker_val
            captures.append((score, move))
    captures.sort(key=lambda x: x[0], reverse=True)

    for _, move in captures[:3]:
        uci = move.uci()
        candidates.append(CandidateMoveRecord(
            uci=uci, san=_san(board, uci), source="heuristic",
        ))
        seen_ucis.add(uci)

    # Priority 2: checking moves
    for move in board.legal_moves:
        uci = move.uci()
        if uci in seen_ucis:
            continue
        if board.gives_check(move):
            candidates.append(CandidateMoveRecord(
                uci=uci, san=_san(board, uci), source="heuristic",
            ))
            seen_ucis.add(uci)
            if len(candidates) >= target_n - 1:
                break

    # Priority 3: central pawn moves and castling
    central_files = {2, 3, 4, 5}  # c, d, e, f files
    for move in board.legal_moves:
        uci = move.uci()
        if uci in seen_ucis:
            continue
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        is_castle = board.is_castling(move)
        is_central_pawn = (
            piece.piece_type == chess.PAWN
            and chess.square_file(move.to_square) in central_files
        )
        if is_castle or is_central_pawn:
            candidates.append(CandidateMoveRecord(
                uci=uci, san=_san(board, uci), source="heuristic",
            ))
            seen_ucis.add(uci)
            if len(candidates) >= target_n - 1:
                break

    # Fill remaining slots with random legal moves
    legal_remaining = [m.uci() for m in board.legal_moves if m.uci() not in seen_ucis]
    rng.shuffle(legal_remaining)
    for uci in legal_remaining[: max(0, target_n - len(candidates))]:
        candidates.append(CandidateMoveRecord(
            uci=uci, san=_san(board, uci), source="random",
        ))

    return candidates

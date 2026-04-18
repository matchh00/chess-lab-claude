from __future__ import annotations

import chess

from src.storage.models import CandidateMoveRecord


def filter_candidates(
    candidates: list[CandidateMoveRecord],
    board: chess.Board,
) -> list[CandidateMoveRecord]:
    """Remove illegal moves and duplicates. Preserve source order for legal, unique entries."""
    seen: set[str] = set()
    legal_ucis = {m.uci() for m in board.legal_moves}
    filtered: list[CandidateMoveRecord] = []

    for c in candidates:
        if c.uci in seen:
            continue
        if c.uci not in legal_ucis:
            continue
        seen.add(c.uci)
        filtered.append(c)

    return filtered

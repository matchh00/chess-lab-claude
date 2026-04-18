from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Optional

import chess

from src.environment.board_manager import BoardManager
from src.environment.engine_wrapper import EngineWrapper
from src.environment.opponents import BaseOpponent
from src.storage.models import MoveRecord

logger = logging.getLogger(__name__)


class MoveLoop:
    """Executes a single move for either the lab side or the opponent."""

    def __init__(
        self,
        board_manager: BoardManager,
        lab_color: chess.Color,
        opponent: BaseOpponent,
        engine: Optional[EngineWrapper] = None,
    ):
        self.bm = board_manager
        self.lab_color = lab_color
        self.opponent = opponent
        self.engine = engine

    def _get_eval(self) -> Optional[float]:
        if self.engine is None:
            return None
        try:
            return self.engine.evaluate(self.bm.board)
        except Exception as e:
            logger.warning("Engine eval failed: %s", e)
            return None

    def step(self) -> Optional[MoveRecord]:
        """Execute one half-move. Returns MoveRecord or None if game over."""
        if self.bm.is_game_over:
            return None

        fen_before = self.bm.fen
        ply = self.bm.ply_index
        is_lab_turn = self.bm.board.turn == self.lab_color
        player_label = "self" if is_lab_turn else "opponent"

        eval_before = self._get_eval()

        if is_lab_turn:
            # Phase 1: lab uses a simple heuristic (will be replaced by LLM in Phase 6)
            uci = self._lab_move()
        else:
            uci = self.opponent.choose_move(self.bm.board)

        san = self.bm.uci_to_san(uci)
        self.bm.push_uci(uci)
        fen_after = self.bm.fen

        eval_after = self._get_eval()

        centipawn_loss: Optional[float] = None
        if eval_before is not None and eval_after is not None:
            # eval_after is from new side-to-move perspective; negate to compare
            centipawn_loss = max(0.0, eval_before - (-eval_after))

        record = MoveRecord(
            ply_index=ply,
            uci=uci,
            san=san,
            fen_before=fen_before,
            fen_after=fen_after,
            player=player_label,
            engine_eval_before=eval_before,
            engine_eval_after=eval_after,
            centipawn_loss=centipawn_loss,
            timestamp=datetime.now(UTC),
        )

        logger.debug("Ply %d [%s] %s  eval_before=%.1f", ply, player_label, san,
                     eval_before if eval_before is not None else 0)
        return record

    def _lab_move(self) -> str:
        """Simple placeholder: heuristic capture preference."""
        board = self.bm.board
        best_score = -999999
        best_move = None
        import random
        for move in board.legal_moves:
            score = 0
            captured = board.piece_at(move.to_square)
            if captured:
                score += 10
            if move.promotion:
                score += 50
            score += random.uniform(0, 0.1)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move.uci() if best_move else next(iter(board.legal_moves)).uci()

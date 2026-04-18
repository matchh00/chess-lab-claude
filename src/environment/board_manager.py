from __future__ import annotations

import uuid
from typing import Optional

import chess
import chess.pgn

from src.storage.models import BoardSnapshot


class BoardManager:
    def __init__(self, fen: Optional[str] = None, game_id: Optional[str] = None):
        self.game_id = game_id or str(uuid.uuid4())
        self.board = chess.Board(fen) if fen else chess.Board()
        self._move_history: list[chess.Move] = []

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    @property
    def ply_index(self) -> int:
        return self.board.ply()

    @property
    def is_check(self) -> bool:
        return self.board.is_check()

    @property
    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    @property
    def result(self) -> Optional[str]:
        if self.board.is_game_over():
            return self.board.result()
        return None

    @property
    def termination(self) -> Optional[str]:
        outcome = self.board.outcome()
        if outcome is None:
            return None
        return outcome.termination.name.lower()

    def legal_moves_uci(self) -> list[str]:
        return [m.uci() for m in self.board.legal_moves]

    def move_history_uci(self) -> list[str]:
        return [m.uci() for m in self._move_history]

    def push_uci(self, uci: str) -> chess.Move:
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci} in position {self.fen}")
        self._move_history.append(move)
        self.board.push(move)
        return move

    def push_san(self, san: str) -> chess.Move:
        move = self.board.parse_san(san)
        self._move_history.append(move)
        self.board.push(move)
        return move

    def uci_to_san(self, uci: str) -> str:
        move = chess.Move.from_uci(uci)
        return self.board.san(move)

    def san_to_uci(self, san: str) -> str:
        return self.board.parse_san(san).uci()

    def snapshot(self) -> BoardSnapshot:
        return BoardSnapshot(
            game_id=self.game_id,
            ply_index=self.ply_index,
            fen=self.fen,
            turn=self.turn,
            legal_moves=self.legal_moves_uci(),
            move_history=self.move_history_uci(),
            is_check=self.is_check,
            is_game_over=self.is_game_over,
            result_if_terminal=self.result,
        )

    def copy(self) -> BoardManager:
        new_mgr = BoardManager(game_id=self.game_id)
        new_mgr.board = self.board.copy()
        new_mgr._move_history = list(self._move_history)
        return new_mgr

    def reset(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen) if fen else chess.Board()
        self._move_history = []

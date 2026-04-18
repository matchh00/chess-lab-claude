from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import chess

from src.environment.engine_wrapper import EngineWrapper


class BaseOpponent(ABC):
    name: str = "base"

    @abstractmethod
    def choose_move(self, board: chess.Board) -> str:
        """Return a UCI move string."""

    def on_game_start(self) -> None:
        pass

    def on_game_end(self) -> None:
        pass


class RandomOpponent(BaseOpponent):
    name = "random_bot"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> str:
        moves = list(board.legal_moves)
        return self._rng.choice(moves).uci()


class HeuristicOpponent(BaseOpponent):
    """Simple heuristic: prefer captures > checks > other moves."""

    name = "heuristic_bot"

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _score_move(self, board: chess.Board, move: chess.Move) -> float:
        score = 0.0
        # Captures
        captured = board.piece_at(move.to_square)
        if captured:
            score += 10 * self.PIECE_VALUES.get(captured.piece_type, 0)
        # En passant capture
        if board.is_en_passant(move):
            score += 10
        # Promotions
        if move.promotion:
            score += 10 * self.PIECE_VALUES.get(move.promotion, 0)
        # Add tiny noise to break ties
        score += self._rng.uniform(0, 0.1)
        return score

    def choose_move(self, board: chess.Board) -> str:
        moves = list(board.legal_moves)
        scored = [(self._score_move(board, m), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1].uci()


class StockfishOpponent(BaseOpponent):
    name = "stockfish"

    def __init__(self, engine: EngineWrapper):
        self._engine = engine

    def choose_move(self, board: chess.Board) -> str:
        moves = self._engine.best_move(board, top_n=1)
        if moves:
            return moves[0]
        # Fallback: random legal move
        return random.choice(list(board.legal_moves)).uci()

    def on_game_start(self) -> None:
        self._engine.open()

    def on_game_end(self) -> None:
        self._engine.close()


def make_opponent(opponent_type: str, engine_path: Optional[str] = None,
                  skill_level: int = 10, depth: int = 10,
                  time_limit: float = 0.1, seed: Optional[int] = None) -> BaseOpponent:
    if opponent_type == "random":
        return RandomOpponent(seed=seed)
    elif opponent_type == "heuristic":
        return HeuristicOpponent(seed=seed)
    elif opponent_type == "stockfish":
        if engine_path is None:
            raise ValueError("engine_path required for stockfish opponent")
        engine = EngineWrapper(engine_path, skill_level=skill_level, depth=depth, time_limit=time_limit)
        return StockfishOpponent(engine)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

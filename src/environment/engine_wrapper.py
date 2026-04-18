from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import chess
import chess.engine


class EngineWrapper:
    def __init__(self, path: str, skill_level: int = 10, depth: int = 10, time_limit: float = 0.1):
        self.path = path
        self.skill_level = skill_level
        self.depth = depth
        self.time_limit = time_limit
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def open(self) -> None:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Skill Level": self.skill_level})

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> EngineWrapper:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def best_move(self, board: chess.Board, top_n: int = 1) -> list[str]:
        """Return top N moves as UCI strings."""
        assert self._engine is not None, "Engine not opened"
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        result = self._engine.analyse(
            board,
            limit,
            multipv=top_n,
        )
        if isinstance(result, list):
            moves = []
            for info in result:
                pv = info.get("pv")
                if pv:
                    moves.append(pv[0].uci())
            return moves
        else:
            pv = result.get("pv")
            if pv:
                return [pv[0].uci()]
        return []

    def analyse_top_n(self, board: chess.Board, n: int = 5) -> list[tuple[str, Optional[float]]]:
        """Return list of (uci, centipawn_eval) for the top N moves from side-to-move perspective."""
        assert self._engine is not None, "Engine not opened"
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        result = self._engine.analyse(board, limit, multipv=n)
        entries = result if isinstance(result, list) else [result]
        output: list[tuple[str, Optional[float]]] = []
        for info in entries:
            pv = info.get("pv")
            if not pv:
                continue
            uci = pv[0].uci()
            score = info.get("score")
            cp: Optional[float] = None
            if score is not None:
                pov = score.pov(board.turn)
                if pov.is_mate():
                    cp = 10000.0 if (pov.mate() is not None and pov.mate() > 0) else -10000.0
                else:
                    raw = pov.score()
                    cp = float(raw) if raw is not None else None
            output.append((uci, cp))
        return output

    def evaluate(self, board: chess.Board) -> Optional[float]:
        """Return centipawn evaluation from the perspective of the side to move."""
        assert self._engine is not None, "Engine not opened"
        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        info = self._engine.analyse(board, limit)
        score = info.get("score")
        if score is None:
            return None
        pov_score = score.pov(board.turn)
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            return 10000.0 if (mate_in is not None and mate_in > 0) else -10000.0
        cp = pov_score.score()
        return float(cp) if cp is not None else None

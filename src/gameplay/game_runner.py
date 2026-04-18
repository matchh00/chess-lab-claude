from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

import chess
import yaml

from src.environment.board_manager import BoardManager
from src.environment.engine_wrapper import EngineWrapper
from src.environment.opponents import BaseOpponent, make_opponent
from src.gameplay.move_loop import MoveLoop
from src.storage.models import GameTrace

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


class GameRunner:
    def __init__(
        self,
        config: dict[str, Any],
        opponent: Optional[BaseOpponent] = None,
        engine: Optional[EngineWrapper] = None,
        lab_color: chess.Color = chess.WHITE,
        game_id: Optional[str] = None,
        initial_fen: Optional[str] = None,
    ):
        self.config = config
        self.lab_color = lab_color
        self.game_id = game_id or str(uuid.uuid4())
        self.initial_fen = initial_fen
        self._engine = engine
        self._owns_engine = False

        if opponent is None:
            self._opponent = self._make_default_opponent()
        else:
            self._opponent = opponent

        self._engine_eval_enabled = engine is not None

    def _make_default_opponent(self) -> BaseOpponent:
        opp_type = self.config.get("default_opponent", "random")
        if opp_type == "stockfish":
            engine_path = self.config.get("stockfish_path", "/opt/homebrew/bin/stockfish")
            skill = self.config.get("stockfish_skill_level", 5)
            depth = self.config.get("stockfish_depth", 10)
            time_limit = self.config.get("stockfish_time_limit", 0.1)
            return make_opponent("stockfish", engine_path=engine_path,
                                 skill_level=skill, depth=depth, time_limit=time_limit)
        return make_opponent(opp_type)

    def run(self) -> GameTrace:
        max_moves = self.config.get("game", {}).get("max_moves", 200)
        white_name = "lab" if self.lab_color == chess.WHITE else self._opponent.name
        black_name = "lab" if self.lab_color == chess.BLACK else self._opponent.name

        trace = GameTrace(
            game_id=self.game_id,
            start_time=datetime.now(UTC),
            config=self.config,
            white_player=white_name,
            black_player=black_name,
            initial_fen=self.initial_fen or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )

        bm = BoardManager(fen=self.initial_fen, game_id=self.game_id)

        self._opponent.on_game_start()
        if self._engine is not None:
            self._engine.open()

        loop = MoveLoop(
            board_manager=bm,
            lab_color=self.lab_color,
            opponent=self._opponent,
            engine=self._engine,
        )

        try:
            for _ in range(max_moves * 2):
                if bm.is_game_over:
                    break
                record = loop.step()
                if record is not None:
                    trace.moves.append(record)
        finally:
            self._opponent.on_game_end()
            if self._engine is not None and self._owns_engine:
                self._engine.close()

        trace.end_time = datetime.now(UTC)
        trace.result = bm.result
        trace.termination = bm.termination
        trace.total_plies = len(trace.moves)

        # Compute average centipawn loss per side
        white_losses = [
            m.centipawn_loss for m in trace.moves
            if m.centipawn_loss is not None and (m.ply_index % 2 == 0)
        ]
        black_losses = [
            m.centipawn_loss for m in trace.moves
            if m.centipawn_loss is not None and (m.ply_index % 2 == 1)
        ]
        if white_losses:
            trace.average_centipawn_loss_white = sum(white_losses) / len(white_losses)
        if black_losses:
            trace.average_centipawn_loss_black = sum(black_losses) / len(black_losses)

        logger.info("Game %s finished: %s (%s), %d plies", self.game_id,
                    trace.result, trace.termination, trace.total_plies)

        self._maybe_save(trace)
        return trace

    def _maybe_save(self, trace: GameTrace) -> None:
        save = self.config.get("game", {}).get("save_traces", False)
        if not save:
            return
        traces_dir = self.config.get("game", {}).get("traces_dir", "data/games")
        Path(traces_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(traces_dir, f"{trace.game_id}.json")
        trace.save(path)
        logger.info("Saved game trace to %s", path)


def run_game_from_config(config_path: str, **kwargs) -> GameTrace:
    config = load_config(config_path)
    runner = GameRunner(config=config, **kwargs)
    return runner.run()

"""Structured log of weight adjustments across games in a learning run."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.learning.rebalancer import WeightAdjustment


class GameAdjustmentEntry(BaseModel):
    game_num: int
    game_id: str
    ply_count: int
    good_move_count: int
    bad_move_count: int
    adjustments: list[WeightAdjustment]


class LearningLog(BaseModel):
    run_id: str
    entries: list[GameAdjustmentEntry] = Field(default_factory=list)

    def add(self, entry: GameAdjustmentEntry) -> None:
        self.entries.append(entry)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

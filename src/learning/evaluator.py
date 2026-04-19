"""Post-move evaluation — derives quality from centipawn loss stored in MoveTrace."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from src.storage.models import MoveTrace


class MoveEvaluation(BaseModel):
    ply_index: int
    centipawn_loss: Optional[float]
    quality_label: Optional[str]  # "good" | "inaccuracy" | "mistake" | "blunder" | None
    selected_rank: Optional[int]  # internal rank chosen (1 = best)
    rank_1_selected: bool


def evaluate_move(trace: MoveTrace) -> MoveEvaluation:
    cpl = trace.centipawn_loss
    label: Optional[str] = None
    if cpl is not None:
        if cpl < 30:
            label = "good"
        elif cpl < 50:
            label = "inaccuracy"
        elif cpl < 100:
            label = "mistake"
        else:
            label = "blunder"

    selected_rank: Optional[int] = None
    rank_1_selected = False
    if trace.decision_record is not None:
        selected_rank = trace.decision_record.selected_internal_rank
        rank_1_selected = selected_rank == 1

    return MoveEvaluation(
        ply_index=trace.ply_index,
        centipawn_loss=cpl,
        quality_label=label,
        selected_rank=selected_rank,
        rank_1_selected=rank_1_selected,
    )

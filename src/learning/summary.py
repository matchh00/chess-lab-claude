"""Learning run summary — per-game performance table and primitive adjustment overview."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.learning.influence import InfluenceRecord
from src.learning.rebalancer import WeightAdjustment
from src.storage.models import MoveTrace


class PerformanceRow(BaseModel):
    game_num: int
    game_id: str
    avg_cpl: Optional[float]
    blunder_count: int
    rank_1_rate: Optional[float]  # fraction 0–1


class PrimitiveAdjustmentSummary(BaseModel):
    bare_key: str
    initial_weight: float
    final_weight: float
    total_delta: float
    net_influence: float


class LearningGameSummary(BaseModel):
    run_id: str
    performance_rows: list[PerformanceRow] = Field(default_factory=list)
    primitive_summaries: list[PrimitiveAdjustmentSummary] = Field(default_factory=list)


def build_game_summary(
    run_id: str,
    game_num: int,
    game_id: str,
    move_traces: list[MoveTrace],
    influence_records: list[InfluenceRecord],
) -> PerformanceRow:
    cpls = [t.centipawn_loss for t in move_traces if t.centipawn_loss is not None]
    avg_cpl = sum(cpls) / len(cpls) if cpls else None
    blunders = sum(1 for t in move_traces if t.blunder_label == "blunder")

    decisions = [t for t in move_traces if t.decision_record is not None]
    rank_1_rate: Optional[float] = None
    if decisions:
        rank_1_rate = sum(1 for t in decisions if t.decision_record.selected_internal_rank == 1) / len(decisions)

    return PerformanceRow(
        game_num=game_num,
        game_id=game_id,
        avg_cpl=round(avg_cpl, 1) if avg_cpl is not None else None,
        blunder_count=blunders,
        rank_1_rate=round(rank_1_rate, 3) if rank_1_rate is not None else None,
    )


def build_primitive_summaries(
    initial_weights: dict[str, float],
    final_weights: dict[str, float],
    all_adjustments: list[WeightAdjustment],
) -> list[PrimitiveAdjustmentSummary]:
    net_by_key: dict[str, float] = {}
    for adj in all_adjustments:
        net_by_key[adj.bare_key] = net_by_key.get(adj.bare_key, 0.0) + adj.net_influence

    summaries: list[PrimitiveAdjustmentSummary] = []
    for key in sorted(initial_weights):
        if key not in final_weights:
            continue
        summaries.append(PrimitiveAdjustmentSummary(
            bare_key=key,
            initial_weight=initial_weights[key],
            final_weight=round(final_weights[key], 4),
            total_delta=round(final_weights[key] - initial_weights[key], 4),
            net_influence=net_by_key.get(key, 0.0),
        ))
    return summaries

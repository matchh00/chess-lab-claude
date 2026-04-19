"""Primitive influence scoring — which primitives were active when moves were good or bad."""
from __future__ import annotations

from pydantic import BaseModel

from src.learning.evaluator import MoveEvaluation
from src.storage.models import MoveTrace


class PrimitiveInfluenceEntry(BaseModel):
    primitive_id: str     # full id with prefix, e.g. "self_material_difference"
    bare_key: str         # weight_map key, e.g. "material_difference"
    category: str
    weighted_score: float


class InfluenceRecord(BaseModel):
    ply_index: int
    quality_label: str | None
    top_primitives: list[PrimitiveInfluenceEntry]
    is_good: bool    # cpl < good_threshold
    is_bad: bool     # cpl > bad_threshold


def _bare_key(primitive_id: str) -> str:
    """Strip 'self_' / 'opponent_' prefix to match PolicyProfile.weight_map keys."""
    for prefix in ("self_", "opponent_"):
        if primitive_id.startswith(prefix):
            return primitive_id[len(prefix):]
    return primitive_id


def score_move_influence(
    trace: MoveTrace,
    evaluation: MoveEvaluation,
    top_n: int = 5,
    good_threshold: float = 30.0,
    bad_threshold: float = 100.0,
) -> InfluenceRecord:
    sorted_states = sorted(trace.weighted_state, key=lambda s: s.weighted_score, reverse=True)
    top = sorted_states[:top_n]

    entries = [
        PrimitiveInfluenceEntry(
            primitive_id=s.primitive_id,
            bare_key=_bare_key(s.primitive_id),
            category=s.category,
            weighted_score=s.weighted_score,
        )
        for s in top
    ]

    cpl = evaluation.centipawn_loss
    is_good = cpl is not None and cpl < good_threshold
    is_bad = cpl is not None and cpl > bad_threshold

    return InfluenceRecord(
        ply_index=trace.ply_index,
        quality_label=evaluation.quality_label,
        top_primitives=entries,
        is_good=is_good,
        is_bad=is_bad,
    )

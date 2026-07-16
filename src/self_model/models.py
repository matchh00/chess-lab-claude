from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MoveReflection(BaseModel):
    """One lab decision and its measured outcome — raw material for the self-model."""

    game_id: str
    ply_index: int
    san: str
    uci: str
    confidence: float = Field(ge=0.0, le=1.0)
    centipawn_loss: Optional[float] = None
    blunder_label: Optional[str] = None  # "blunder" | "mistake" | "inaccuracy" | None
    reasoning_summary: str = ""
    dominant_categories: list[str] = Field(default_factory=list)
    fallback_used: bool = False

    def to_json(self) -> str:
        return self.model_dump_json()


class CategoryBias(BaseModel):
    """How the lab performs when a given primitive category dominates its signals."""

    category: str
    decision_count: int
    mean_cpl: Optional[float] = None
    blunder_count: int = 0

    def to_json(self) -> str:
        return self.model_dump_json()


class SelfPattern(BaseModel):
    """A detected regularity in the lab's own decision-making.

    kind identifies the detector that produced it, so renderers can map
    patterns to directives without parsing text.
    """

    kind: Literal["overconfidence", "calibration_gap", "category_bias", "cost_streak"]
    text: str

    def to_json(self) -> str:
        return self.model_dump_json()


class SelfModelState(BaseModel):
    """Snapshot of what the lab currently believes about its own judgment."""

    scope: Literal["game", "run"] = "game"
    moves_observed: int = 0
    mean_confidence: Optional[float] = None
    mean_cpl: Optional[float] = None
    blunder_count: int = 0
    high_confidence_costly_moves: int = 0
    mean_confidence_on_costly: Optional[float] = None
    mean_confidence_on_clean: Optional[float] = None
    category_biases: list[CategoryBias] = Field(default_factory=list)
    patterns: list[SelfPattern] = Field(default_factory=list)
    reflections: list[MoveReflection] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> SelfModelState:
        return cls.model_validate_json(data)

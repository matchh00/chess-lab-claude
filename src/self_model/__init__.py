"""Self-model experiment (Phase 9).

Three conditions test whether a causally embedded self-model improves
decision quality (see RESEARCH_LOG.md, "Self-model experiment"):

  A (off)     — board model only: the existing v1.1 pipeline.
  B (history) — the prompt additionally shows the lab's recent decisions,
                confidences, and measured outcomes.
  C (full)    — the prompt additionally shows a derived self-model:
                calibration statistics, characteristic-mistake patterns,
                and directives generated from them.

The self-model is only interesting if it is causally load-bearing:
removing it (running A or B instead of C) is the ablation.
"""
from src.self_model.memory import SelfModelMemory
from src.self_model.models import (
    CategoryBias,
    MoveReflection,
    SelfModelState,
    SelfPattern,
)
from src.self_model.renderers import render_history_block, render_self_model_block

__all__ = [
    "CategoryBias",
    "MoveReflection",
    "SelfModelMemory",
    "SelfModelState",
    "SelfPattern",
    "render_history_block",
    "render_self_model_block",
]

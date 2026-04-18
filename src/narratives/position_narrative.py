from __future__ import annotations

from src.narratives.token_budget import BudgetResult, check_budget
from src.policies.profiles import PolicyProfile
from src.storage.models import PrimitiveValue, WeightedPrimitiveState


def build_position_narrative(
    weighted_states: list[WeightedPrimitiveState],
    primitives: list[PrimitiveValue],
    policy: PolicyProfile,
    word_limit: int = 150,
) -> BudgetResult:
    """Build a human-readable position narrative from weighted primitive states.

    Selects top positive signals (norm>=0.65, score>=0.5) and top negative signals
    (norm<0.35, weight>=1.0), renders their text_render strings, appends a policy
    directive, then enforces the word limit.
    """
    primitive_map: dict[str, PrimitiveValue] = {p.primitive_id: p for p in primitives}

    sorted_by_score = sorted(weighted_states, key=lambda s: s.weighted_score, reverse=True)

    positive_signals: list[str] = []
    for state in sorted_by_score:
        if state.normalized_value >= 0.65 and state.weighted_score >= 0.5:
            prim = primitive_map.get(state.primitive_id)
            if prim and prim.text_render:
                positive_signals.append(prim.text_render)
        if len(positive_signals) >= 5:
            break

    negative_signals: list[str] = []
    for state in sorted(weighted_states, key=lambda s: s.normalized_value):
        if state.normalized_value < 0.35 and state.weight >= 1.0:
            prim = primitive_map.get(state.primitive_id)
            if prim and prim.text_render:
                negative_signals.append(prim.text_render)
        if len(negative_signals) >= 3:
            break

    parts: list[str] = []
    if positive_signals:
        parts.append("Strengths: " + " ".join(positive_signals))
    if negative_signals:
        parts.append("Concerns: " + " ".join(negative_signals))

    style = getattr(policy, "text_priority_style", "balanced")
    directive_map = {
        "aggressive": "Prioritize active play, threats, and initiative.",
        "defensive": "Prioritize king safety and solid structure.",
        "development_first": "Prioritize piece development and central control.",
        "endgame_clean": "Prioritize pawn promotion paths and king activity.",
        "balanced": "Balance positional and tactical considerations.",
    }
    directive = directive_map.get(style, "Balance positional and tactical considerations.")
    parts.append(directive)

    full_text = " ".join(parts).strip()
    return check_budget(full_text, word_limit, budget_type="words")

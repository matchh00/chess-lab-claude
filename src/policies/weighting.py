from __future__ import annotations

from src.policies.profiles import PolicyProfile
from src.storage.models import PrimitiveValue, WeightedPrimitiveState


_IMPORTANCE_THRESHOLDS: list[tuple[float, str]] = [
    (1.5, "critical"),
    (1.0, "high"),
    (0.5, "medium"),
    (0.2, "low"),
    (0.0, "negligible"),
]

_IMPORTANCE_MESSAGES: dict[str, str] = {
    "critical": "critical priority — address immediately",
    "high":     "high priority — weight heavily in decisions",
    "medium":   "moderate priority — consider alongside other factors",
    "low":      "low priority — minor factor",
    "negligible": "negligible — not a deciding factor",
}


def _importance_label(weighted_score: float) -> str:
    for threshold, label in _IMPORTANCE_THRESHOLDS:
        if weighted_score >= threshold:
            return label
    return "negligible"


def apply_policy(
    primitives: list[PrimitiveValue],
    policy: PolicyProfile,
) -> list[WeightedPrimitiveState]:
    """Apply policy weights to a list of PrimitiveValues.

    effective_score = normalized_value * weight * confidence
    """
    states: list[WeightedPrimitiveState] = []
    for pv in primitives:
        weight = policy.get_weight(pv.primitive_id, pv.category)
        weighted_score = pv.normalized_value * weight * pv.confidence
        label = _importance_label(weighted_score)
        states.append(WeightedPrimitiveState(
            primitive_id=pv.primitive_id,
            name=pv.name,
            category=pv.category,
            raw_value=pv.value,
            normalized_value=pv.normalized_value,
            weight=weight,
            confidence=pv.confidence,
            weighted_score=round(weighted_score, 4),
            importance_label=label,
            policy_message=f"{pv.name}: {_IMPORTANCE_MESSAGES[label]}.",
        ))
    return states


def weighted_scores_by_id(states: list[WeightedPrimitiveState]) -> dict[str, float]:
    return {s.primitive_id: s.weighted_score for s in states}

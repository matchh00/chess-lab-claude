from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.policies.profiles import PolicyProfile
from src.storage.models import WeightedPrimitiveState


# Per-style, per-category plain-language sentences.
# Loaded at runtime; never hardcoded weights — this is text rendering only.
_CATEGORY_SENTENCES: dict[str, dict[str, str]] = {
    "moderate": {
        "material":       "Maintain material balance and avoid unnecessary piece loss.",
        "king_safety":    "Keep your king protected behind a solid pawn shield.",
        "development":    "Complete development efficiently and consider castling.",
        "center_control": "Contest the center with pawns and active pieces.",
        "tactical":       "Stay alert to tactical opportunities and threats.",
        "piece_activity": "Activate your pieces to strong squares.",
        "pawn_structure": "Preserve a healthy, cohesive pawn structure.",
        "initiative":     "Maintain initiative when it can be safely preserved.",
    },
    "aggressive": {
        "material":       "Accept material imbalances when attacking chances are gained.",
        "king_safety":    "Trade safety for momentum only when the attack is concrete.",
        "development":    "Develop aggressively with tempo and immediate threats.",
        "center_control": "Seize the center with maximum force.",
        "tactical":       "Create and exploit tactical complications at every turn.",
        "piece_activity": "Maximize piece activity; push pieces to attacking posts.",
        "pawn_structure": "Sacrifice pawn structure for dynamic piece play.",
        "initiative":     "Seize and escalate initiative at every opportunity.",
    },
    "defensive": {
        "material":       "Protect your material carefully; avoid dropped pieces.",
        "king_safety":    "Prioritize king safety above all other considerations.",
        "development":    "Develop solidly and castle as soon as possible.",
        "center_control": "Maintain a solid, defensive central presence.",
        "tactical":       "Neutralize opponent threats before considering your own.",
        "piece_activity": "Coordinate pieces defensively; avoid overextension.",
        "pawn_structure": "Keep a strong, unbroken pawn structure.",
        "initiative":     "Cede initiative only when it safely neutralizes threats.",
    },
    "developmental": {
        "material":       "Minor material investment is acceptable for a development lead.",
        "king_safety":    "Castle early to secure the king and connect rooks.",
        "development":    "Develop all minor pieces before launching any attack.",
        "center_control": "Establish central pawn presence and piece influence.",
        "tactical":       "Use tactics primarily to gain tempo in development.",
        "piece_activity": "Place all pieces on their ideal squares rapidly.",
        "pawn_structure": "Keep pawns flexible to support active piece play.",
        "initiative":     "Gain initiative through superior development speed.",
    },
    "endgame": {
        "material":       "Convert material advantages precisely; simplify cleanly.",
        "king_safety":    "Activate your king — it is a powerful endgame piece.",
        "development":    "Prioritize piece coordination over raw development.",
        "center_control": "Centralize all pieces; control key squares and open files.",
        "tactical":       "Spot precise combinations to convert the advantage.",
        "piece_activity": "Keep rooks active on open files and the seventh rank.",
        "pawn_structure": "Passed pawns are your most valuable asset — advance them.",
        "initiative":     "Maintain active play; passivity concedes the conversion.",
    },
}

_FALLBACK_STYLE = "moderate"


def machine_summary(states: list[WeightedPrimitiveState]) -> list[dict]:
    """Return primitives sorted by weighted_score descending, as plain dicts."""
    sorted_states = sorted(states, key=lambda s: s.weighted_score, reverse=True)
    return [s.model_dump() for s in sorted_states]


def text_priority_summary(
    policy: PolicyProfile,
    states: list[WeightedPrimitiveState],
) -> str:
    """Produce a human-readable policy priority paragraph from weighted states.

    Ranks categories by (group_weight × average_weighted_score), then emits
    style-specific sentences for the top categories.
    """
    # 1. Average weighted score per category
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for s in states:
        cat_scores[s.category].append(s.weighted_score)

    cat_avg: dict[str, float] = {
        cat: sum(scores) / len(scores)
        for cat, scores in cat_scores.items()
    }

    # 2. Multiply by group_weight to get priority rank
    cat_priority: dict[str, float] = {
        cat: avg * policy.group_weights.get(cat, 1.0)
        for cat, avg in cat_avg.items()
    }

    sorted_cats = sorted(cat_priority, key=lambda c: cat_priority[c], reverse=True)

    # 3. Pick style sentences for top 5 categories
    style = policy.text_priority_style
    sentences_map = _CATEGORY_SENTENCES.get(style, _CATEGORY_SENTENCES[_FALLBACK_STYLE])
    sentences = [
        sentences_map.get(cat, f"Attend to {cat.replace('_', ' ')}.")
        for cat in sorted_cats[:5]
    ]

    return " ".join(sentences)


def top_primitives_text(states: list[WeightedPrimitiveState], n: int = 5) -> str:
    """Return a bullet list of the top-N primitives by weighted_score."""
    top = sorted(states, key=lambda s: s.weighted_score, reverse=True)[:n]
    lines = [
        f"  [{s.importance_label.upper()}] {s.policy_message} (score={s.weighted_score:.3f})"
        for s in top
    ]
    return "\n".join(lines)

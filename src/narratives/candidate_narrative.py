from __future__ import annotations

from src.narratives.token_budget import BudgetResult, check_budget
from src.storage.models import CandidateMoveRecord

_FLAG_DESCRIPTIONS: dict[str, str] = {
    "hangs_piece": "risks leaving a piece hanging",
    "exposes_king": "weakens the king's pawn shield",
    "loses_material": "leads to material loss",
    "king_under_pressure": "invites enemy attackers toward your king",
}


def build_candidate_narrative(
    candidate: CandidateMoveRecord,
    word_limit: int = 60,
) -> BudgetResult:
    """Build a short narrative for a single candidate move."""
    parts: list[str] = []

    if candidate.engine_eval_after is not None:
        eval_cp = candidate.engine_eval_after
        if eval_cp > 100:
            parts.append("Engine strongly favors this move.")
        elif eval_cp > 30:
            parts.append("Engine slightly favors this move.")
        elif eval_cp < -100:
            parts.append("Engine rates this move poorly.")
        elif eval_cp < -30:
            parts.append("Engine is slightly negative on this move.")
        else:
            parts.append("Engine considers this roughly equal.")

    if candidate.primitive_delta_summary and candidate.primitive_delta_summary != "No significant primitive changes.":
        parts.append(candidate.primitive_delta_summary)

    if candidate.risk_flags:
        flag_phrases = [
            _FLAG_DESCRIPTIONS.get(f, f.replace("_", " "))
            for f in candidate.risk_flags
        ]
        parts.append("Caution: " + "; ".join(flag_phrases) + ".")

    if not parts:
        parts.append("No significant evaluation signals.")

    text = " ".join(parts).strip()
    return check_budget(text, word_limit, budget_type="words")

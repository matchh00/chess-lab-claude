from __future__ import annotations

import chess

from src.policies.profiles import PolicyProfile
from src.policies.weighting import apply_policy, weighted_scores_by_id
from src.primitives.extractor import PrimitiveExtractor
from src.storage.models import CandidateMoveRecord, WeightedPrimitiveState

# Primitives whose scores depend on whose turn it is; skip in delta computation
# to avoid spuriously negative deltas on every candidate.
_TURN_DEPENDENT = {
    "self_checks_available",
    "self_captures_available",
    "self_forcing_moves_count",
    "self_immediate_threat",
    "self_tempo_gaining_candidate",
}

_DELTA_THRESHOLD = 0.05   # min absolute change to report in summary
_DELTA_RISK_THRESH = 0.15  # threshold for risk flags


def _render_delta(
    baseline: dict[str, float],
    after: dict[str, float],
    top_n: int = 3,
) -> str:
    deltas: dict[str, float] = {}
    for pid, after_score in after.items():
        if pid in _TURN_DEPENDENT:
            continue
        delta = after_score - baseline.get(pid, 0.0)
        if abs(delta) >= _DELTA_THRESHOLD:
            deltas[pid] = delta

    improvements = sorted(
        [(pid, d) for pid, d in deltas.items() if d > 0],
        key=lambda x: x[1], reverse=True,
    )
    concerns = sorted(
        [(pid, d) for pid, d in deltas.items() if d < 0],
        key=lambda x: x[1],
    )

    def _short(pid: str) -> str:
        # Strip "self_" prefix for readability
        return pid.replace("self_", "").replace("_", " ")

    parts: list[str] = []
    if improvements[:top_n]:
        items = ", ".join(f"{_short(p)} (+{d:.2f})" for p, d in improvements[:top_n])
        parts.append(f"+: {items}")
    if concerns[:top_n]:
        items = ", ".join(f"{_short(p)} ({d:.2f})" for p, d in concerns[:top_n])
        parts.append(f"-: {items}")

    return " | ".join(parts) if parts else "No significant primitive changes."


def _compute_risk_flags(
    baseline: dict[str, float],
    after: dict[str, float],
) -> list[str]:
    flags: list[str] = []
    hanging_delta = after.get("self_hanging_own_pieces", 0.0) - baseline.get("self_hanging_own_pieces", 0.0)
    if hanging_delta < -_DELTA_RISK_THRESH:
        flags.append("hangs_piece")

    shield_delta = after.get("self_pawn_shield_quality", 0.0) - baseline.get("self_pawn_shield_quality", 0.0)
    if shield_delta < -_DELTA_RISK_THRESH:
        flags.append("exposes_king")

    material_delta = after.get("self_material_difference", 0.0) - baseline.get("self_material_difference", 0.0)
    if material_delta < -_DELTA_RISK_THRESH:
        flags.append("loses_material")

    attackers_delta = after.get("self_enemy_attackers_near_king", 0.0) - baseline.get("self_enemy_attackers_near_king", 0.0)
    if attackers_delta < -_DELTA_RISK_THRESH:
        flags.append("king_under_pressure")

    return flags


def annotate_candidates(
    candidates: list[CandidateMoveRecord],
    board: chess.Board,
    color: chess.Color,
    extractor: PrimitiveExtractor,
    policy: PolicyProfile,
    baseline_states: list[WeightedPrimitiveState],
) -> list[CandidateMoveRecord]:
    """Simulate each candidate, extract primitives, compute deltas, set risk flags.

    Returns a new list of CandidateMoveRecord with annotation fields filled.
    Does not alter internal_rank or presentation_index.
    """
    baseline_scores = weighted_scores_by_id(baseline_states)
    annotated: list[CandidateMoveRecord] = []

    for candidate in candidates:
        move = chess.Move.from_uci(candidate.uci)
        board_copy = board.copy()
        board_copy.push(move)

        board_after_fen = board_copy.fen()

        # Extract primitives from our perspective after the move
        primitives_after = extractor.extract(board_copy, color)
        states_after = apply_policy(primitives_after, policy)
        scores_after = weighted_scores_by_id(states_after)

        delta_summary = _render_delta(baseline_scores, scores_after)
        risk_flags = _compute_risk_flags(baseline_scores, scores_after)

        # Total score delta (structural primitives only)
        structural_ids = {
            pid for pid in scores_after
            if pid not in _TURN_DEPENDENT
        }
        score_delta = sum(
            scores_after.get(pid, 0.0) - baseline_scores.get(pid, 0.0)
            for pid in structural_ids
        )

        annotated.append(candidate.model_copy(update={
            "board_after_fen": board_after_fen,
            "primitive_delta_summary": delta_summary,
            "risk_flags": risk_flags,
            "score_delta": round(score_delta, 4),
            "primitives_after": primitives_after,
        }))

    return annotated

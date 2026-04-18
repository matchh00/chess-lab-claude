from __future__ import annotations

from typing import Optional

import pandas as pd

from src.storage.models import MoveTrace

BLUNDER_THRESHOLD = 200.0
MISTAKE_THRESHOLD = 100.0
INACCURACY_THRESHOLD = 50.0


def compute_blunder_label(centipawn_loss: Optional[float]) -> Optional[str]:
    if centipawn_loss is None:
        return None
    if centipawn_loss > BLUNDER_THRESHOLD:
        return "blunder"
    if centipawn_loss > MISTAKE_THRESHOLD:
        return "mistake"
    if centipawn_loss > INACCURACY_THRESHOLD:
        return "inaccuracy"
    return None


def extract_move_row(trace: MoveTrace) -> dict:
    dr = trace.decision_record
    pr = trace.prompt_record
    return {
        "game_id": trace.game_id,
        "ply_index": trace.ply_index,
        "fen": trace.board_snapshot.fen,
        "centipawn_loss": trace.centipawn_loss,
        "blunder_label": trace.blunder_label,
        "engine_eval_before": trace.engine_eval_before,
        "engine_eval_after": trace.engine_eval_after,
        "selected_internal_rank": dr.selected_internal_rank if dr else None,
        "selected_presentation_index": dr.selected_presentation_index if dr else None,
        "confidence": dr.confidence if dr else None,
        "fallback_used": dr.fallback_used if dr else None,
        "is_valid": dr.is_valid if dr else None,
        "token_count": pr.token_count if pr else trace.token_count,
        "position_narrative_words": trace.position_narrative_word_count,
        "num_candidates": len(trace.candidates),
    }


def build_move_log_df(traces: list[MoveTrace]) -> pd.DataFrame:
    if not traces:
        return pd.DataFrame()
    rows = [extract_move_row(t) for t in traces]
    return pd.DataFrame(rows)

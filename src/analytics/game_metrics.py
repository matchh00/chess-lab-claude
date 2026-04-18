from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from src.analytics.move_metrics import BLUNDER_THRESHOLD, MISTAKE_THRESHOLD, INACCURACY_THRESHOLD
from src.storage.models import GameTrace, MoveTrace


def accuracy_estimate(avg_cp_loss: float) -> float:
    """Chess.com-style accuracy from average centipawn loss."""
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * avg_cp_loss) - 3.1668))


def compute_game_metrics(
    game_trace: GameTrace,
    move_traces: list[MoveTrace],
) -> dict:
    cp_losses = [t.centipawn_loss for t in move_traces if t.centipawn_loss is not None]
    avg_cpl = sum(cp_losses) / len(cp_losses) if cp_losses else None

    blunder_count = sum(1 for l in cp_losses if l > BLUNDER_THRESHOLD)
    mistake_count = sum(1 for l in cp_losses if MISTAKE_THRESHOLD < l <= BLUNDER_THRESHOLD)
    inaccuracy_count = sum(1 for l in cp_losses if INACCURACY_THRESHOLD < l <= MISTAKE_THRESHOLD)

    token_counts = [
        (t.prompt_record.token_count if t.prompt_record else t.token_count)
        for t in move_traces
    ]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    confidences = [
        t.decision_record.confidence
        for t in move_traces
        if t.decision_record and t.decision_record.confidence > 0
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    rank_dist: dict[int, int] = {}
    pidx_dist: dict[int, int] = {}
    for t in move_traces:
        if t.decision_record:
            r = t.decision_record.selected_internal_rank
            rank_dist[r] = rank_dist.get(r, 0) + 1
            p = t.decision_record.selected_presentation_index
            pidx_dist[p] = pidx_dist.get(p, 0) + 1

    result = game_trace.result or "unknown"
    lab_color = "white"  # determined by ply parity (even plies = white)

    return {
        "game_id": game_trace.game_id,
        "result": result,
        "total_plies": game_trace.total_plies,
        "lab_moves": len(move_traces),
        "avg_centipawn_loss": avg_cpl,
        "accuracy_estimate": accuracy_estimate(avg_cpl) if avg_cpl is not None else None,
        "blunder_count": blunder_count,
        "mistake_count": mistake_count,
        "inaccuracy_count": inaccuracy_count,
        "avg_token_count": avg_tokens,
        "total_tokens": sum(token_counts),
        "avg_llm_confidence": avg_confidence,
        "candidate_rank_distribution": rank_dist,
        "presentation_index_distribution": pidx_dist,
        "white_player": game_trace.white_player,
        "black_player": game_trace.black_player,
    }


def build_game_summary_df(metrics_list: list[dict]) -> pd.DataFrame:
    if not metrics_list:
        return pd.DataFrame()
    flat = []
    for m in metrics_list:
        row = {k: v for k, v in m.items()
               if k not in ("candidate_rank_distribution", "presentation_index_distribution")}
        flat.append(row)
    return pd.DataFrame(flat)

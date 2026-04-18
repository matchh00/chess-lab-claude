from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.storage.models import MoveTrace


def compute_run_metrics(
    run_id: str,
    game_summary_df: pd.DataFrame,
    all_move_traces: list[MoveTrace],
) -> dict:
    if game_summary_df.empty:
        return {"run_id": run_id, "total_games": 0}

    n = len(game_summary_df)
    results = game_summary_df["result"].tolist()

    def _is_win(r: str) -> bool:
        # Lab plays white; win = "1-0", loss = "0-1"
        # This is a simplification — needs lab_color to be precise
        return r in ("1-0",)

    def _is_draw(r: str) -> bool:
        return "1/2" in str(r)

    def _is_loss(r: str) -> bool:
        return r in ("0-1",)

    wins = sum(1 for r in results if _is_win(r))
    draws = sum(1 for r in results if _is_draw(r))
    losses = sum(1 for r in results if _is_loss(r))

    cpl_col = game_summary_df["avg_centipawn_loss"].dropna()
    avg_cpl = float(cpl_col.mean()) if not cpl_col.empty else None

    total_tokens = int(game_summary_df["total_tokens"].sum())

    rank_dist: dict[int, int] = defaultdict(int)
    pidx_dist: dict[int, int] = defaultdict(int)
    for t in all_move_traces:
        if t.decision_record:
            rank_dist[t.decision_record.selected_internal_rank] += 1
            pidx_dist[t.decision_record.selected_presentation_index] += 1

    blunder_total = int(game_summary_df["blunder_count"].sum())

    return {
        "run_id": run_id,
        "total_games": n,
        "win_count": wins,
        "draw_count": draws,
        "loss_count": losses,
        "win_rate": wins / n if n else 0.0,
        "draw_rate": draws / n if n else 0.0,
        "loss_rate": losses / n if n else 0.0,
        "average_centipawn_loss": avg_cpl,
        "total_token_usage": total_tokens,
        "blunder_count_total": blunder_total,
        "candidate_rank_distribution": dict(rank_dist),
        "presentation_index_distribution": dict(pidx_dist),
    }

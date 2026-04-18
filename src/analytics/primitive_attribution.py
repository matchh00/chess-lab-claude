from __future__ import annotations

import pandas as pd

from src.storage.models import MoveTrace


def compute_primitive_attribution(all_move_traces: list[MoveTrace]) -> pd.DataFrame:
    """Compute Pearson correlation between each primitive's weighted_score and centipawn_loss.

    Only moves with both non-empty weighted_state and non-None centipawn_loss are included.
    Returns DataFrame sorted by absolute correlation descending.
    """
    rows = []
    for t in all_move_traces:
        if not t.weighted_state or t.centipawn_loss is None:
            continue
        row: dict = {"centipawn_loss": t.centipawn_loss}
        for ws in t.weighted_state:
            row[ws.primitive_id] = ws.weighted_score
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["primitive_id", "correlation", "n_observations"])

    df = pd.DataFrame(rows)
    prim_cols = [c for c in df.columns if c != "centipawn_loss"]

    if not prim_cols or df["centipawn_loss"].nunique() < 2:
        return pd.DataFrame(columns=["primitive_id", "correlation", "n_observations"])

    results = []
    for col in prim_cols:
        series = df[[col, "centipawn_loss"]].dropna()
        n = len(series)
        if n < 3 or series[col].nunique() < 2:
            corr = float("nan")
        else:
            corr = float(series[col].corr(series["centipawn_loss"]))
        results.append({"primitive_id": col, "correlation": corr, "n_observations": n})

    result_df = pd.DataFrame(results)
    result_df["abs_correlation"] = result_df["correlation"].abs()
    result_df = result_df.sort_values("abs_correlation", ascending=False).drop(columns=["abs_correlation"])
    return result_df.reset_index(drop=True)

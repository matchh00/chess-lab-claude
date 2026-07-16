from __future__ import annotations

from typing import Optional

import pandas as pd

from src.analytics.move_metrics import BLUNDER_THRESHOLD, INACCURACY_THRESHOLD
from src.storage.models import MoveTrace

# Confidence bins for the calibration table: [low, high) except the last, which
# is inclusive of 1.0.
CONFIDENCE_BINS: list[tuple[float, float]] = [
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.85),
    (0.85, 1.0),
]


def _scored_rows(traces: list[MoveTrace]) -> list[tuple[float, float]]:
    """(confidence, centipawn_loss) for valid, non-fallback decisions with a measured CPL."""
    rows: list[tuple[float, float]] = []
    for t in traces:
        d = t.decision_record
        if d is None or d.fallback_used or t.centipawn_loss is None:
            continue
        rows.append((d.confidence, t.centipawn_loss))
    return rows


def compute_confidence_calibration(traces: list[MoveTrace]) -> dict:
    """How well the LLM's stated confidence tracks realized move quality.

    A well-calibrated player shows a negative confidence-CPL correlation and a
    lower blunder rate in higher confidence bins. The self-model experiment's
    core calibration question is whether condition C improves these numbers
    over conditions A and B.
    """
    rows = _scored_rows(traces)
    if len(rows) < 2:
        return {
            "n_decisions": len(rows),
            "confidence_cpl_correlation": None,
            "mean_confidence": None,
            "mean_confidence_on_blunders": None,
            "mean_confidence_on_clean_moves": None,
            "bins": [],
        }

    df = pd.DataFrame(rows, columns=["confidence", "cpl"])

    corr = df["confidence"].corr(df["cpl"])
    correlation = float(corr) if pd.notna(corr) else None

    blunders = df[df["cpl"] > BLUNDER_THRESHOLD]
    clean = df[df["cpl"] <= INACCURACY_THRESHOLD]

    bins = []
    for low, high in CONFIDENCE_BINS:
        if high >= 1.0:
            mask = (df["confidence"] >= low) & (df["confidence"] <= high)
        else:
            mask = (df["confidence"] >= low) & (df["confidence"] < high)
        group = df[mask]
        bins.append({
            "confidence_range": f"[{low:.2f}, {high:.2f}{']' if high >= 1.0 else ')'}",
            "n": int(len(group)),
            "mean_cpl": float(group["cpl"].mean()) if len(group) else None,
            "blunder_rate": (
                float((group["cpl"] > BLUNDER_THRESHOLD).mean()) if len(group) else None
            ),
        })

    return {
        "n_decisions": int(len(df)),
        "confidence_cpl_correlation": correlation,
        "mean_confidence": float(df["confidence"].mean()),
        "mean_confidence_on_blunders": (
            float(blunders["confidence"].mean()) if len(blunders) else None
        ),
        "mean_confidence_on_clean_moves": (
            float(clean["confidence"].mean()) if len(clean) else None
        ),
        "bins": bins,
    }


def compute_self_model_usage(traces: list[MoveTrace]) -> dict:
    """How often a self-model block was actually present in the prompt.

    The first lab move of each game has no history yet, so presence below
    100% is expected even in modes 'history' and 'full'.
    """
    with_block = sum(1 for t in traces if t.self_model_block)
    modes = sorted({t.self_model_mode for t in traces}) if traces else []
    mean_block_chars: Optional[float] = None
    if with_block:
        mean_block_chars = sum(len(t.self_model_block) for t in traces if t.self_model_block) / with_block
    return {
        "modes_present": modes,
        "moves_with_self_model_block": with_block,
        "total_lab_moves": len(traces),
        "mean_block_chars": mean_block_chars,
    }

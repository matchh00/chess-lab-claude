from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics.game_metrics import build_game_summary_df, compute_game_metrics
from src.analytics.move_metrics import build_move_log_df
from src.analytics.primitive_attribution import compute_primitive_attribution
from src.analytics.run_metrics import compute_run_metrics
from src.analytics.self_model_metrics import (
    compute_confidence_calibration,
    compute_self_model_usage,
)
from src.storage.models import ExperimentManifest, GameTrace, MoveTrace


def build_reports(
    run_id: str,
    run_dir: Path,
    manifest: ExperimentManifest,
    all_move_traces: list[MoveTrace],
    all_game_traces: list[GameTrace],
    traces_by_game: dict[str, list[MoveTrace]],
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("reports") / "latest").mkdir(parents=True, exist_ok=True)

    # ── Move log ──────────────────────────────────────────────────────────────
    move_df = build_move_log_df(all_move_traces)
    move_log_path = run_dir / "move_log.csv"
    move_df.to_csv(move_log_path, index=False)

    # ── Game summary ──────────────────────────────────────────────────────────
    game_metrics_list = [
        compute_game_metrics(gt, traces_by_game.get(gt.game_id, []))
        for gt in all_game_traces
    ]
    game_df = build_game_summary_df(game_metrics_list)
    game_summary_path = run_dir / "game_summary.csv"
    game_df.to_csv(game_summary_path, index=False)

    # ── Run report JSON ───────────────────────────────────────────────────────
    run_metrics = compute_run_metrics(run_id, game_df, all_move_traces)
    run_metrics["confidence_calibration"] = compute_confidence_calibration(all_move_traces)
    run_metrics["self_model_usage"] = compute_self_model_usage(all_move_traces)
    run_report_path = run_dir / "run_report.json"
    with open(run_report_path, "w") as f:
        json.dump(run_metrics, f, indent=2, default=str)

    # ── Primitive attribution ─────────────────────────────────────────────────
    attr_df = compute_primitive_attribution(all_move_traces)
    attr_path = run_dir / "primitive_attribution.csv"
    attr_df.to_csv(attr_path, index=False)

    # ── Markdown summary ──────────────────────────────────────────────────────
    md = _build_markdown(run_id, manifest, run_metrics, attr_df, game_df)
    md_path = Path("reports") / "latest" / f"{run_id}_summary.md"
    with open(md_path, "w") as f:
        f.write(md)

    return {
        "move_log": str(move_log_path),
        "game_summary": str(game_summary_path),
        "run_report": str(run_report_path),
        "primitive_attribution": str(attr_path),
        "markdown_summary": str(md_path),
    }


def _build_markdown(
    run_id: str,
    manifest: ExperimentManifest,
    run_metrics: dict,
    attr_df: pd.DataFrame,
    game_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# Run Report: {run_id}\n")

    lines.append("## Configuration\n")
    lines.append(f"| Setting | Value |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Policy | {manifest.policy_name} |")
    lines.append(f"| Player type | {manifest.player_type} |")
    lines.append(f"| Narrative mode | {manifest.narrative_mode} |")
    lines.append(f"| Candidate mode | {manifest.candidate_mode} |")
    lines.append(f"| Opponent | {manifest.opponent_type} skill {manifest.opponent_skill} |")
    lines.append(f"| Prompt version | {manifest.prompt_version} |")
    lines.append(f"| Self-model mode | {manifest.self_model_mode} ({manifest.self_model_scope} scope) |")
    lines.append(f"| Games | {manifest.completed_games}/{manifest.total_games} |")
    lines.append(f"| Seed | {manifest.random_seed} |")
    lines.append("")

    lines.append("## Key Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    wr = run_metrics.get("win_rate", 0.0)
    dr = run_metrics.get("draw_rate", 0.0)
    lr = run_metrics.get("loss_rate", 0.0)
    avg_cpl = run_metrics.get("average_centipawn_loss")
    total_tok = run_metrics.get("total_token_usage", 0)
    blunders = run_metrics.get("blunder_count_total", 0)
    lines.append(f"| Win rate | {wr:.1%} |")
    lines.append(f"| Draw rate | {dr:.1%} |")
    lines.append(f"| Loss rate | {lr:.1%} |")
    cpl_str = f"{avg_cpl:.1f}" if avg_cpl is not None else "N/A"
    lines.append(f"| Avg centipawn loss | {cpl_str} |")
    lines.append(f"| Blunder count (total) | {blunders} |")
    lines.append(f"| Total token usage | {total_tok:,} |")
    lines.append("")

    if not game_df.empty and "avg_centipawn_loss" in game_df.columns:
        lines.append("## Per-Game Results\n")
        lines.append("| Game | Result | Avg CPL | Blunders | Tokens |")
        lines.append("|------|--------|---------|----------|--------|")
        for _, row in game_df.iterrows():
            cpl = f"{row['avg_centipawn_loss']:.1f}" if pd.notna(row.get("avg_centipawn_loss")) else "N/A"
            lines.append(
                f"| {row['game_id'][:12]}... | {row['result']} | {cpl} "
                f"| {int(row.get('blunder_count', 0))} | {int(row.get('total_tokens', 0)):,} |"
            )
        lines.append("")

    if not attr_df.empty:
        top10 = attr_df.head(10)
        lines.append("## Primitive Attribution (Top 10 by |Correlation| with CPL)\n")
        lines.append("| Primitive | Correlation | N |")
        lines.append("|-----------|-------------|---|")
        for _, row in top10.iterrows():
            corr = f"{row['correlation']:.3f}" if pd.notna(row["correlation"]) else "N/A"
            lines.append(f"| {row['primitive_id']} | {corr} | {int(row['n_observations'])} |")
        lines.append("")

    calibration = run_metrics.get("confidence_calibration", {})
    if calibration.get("n_decisions", 0) >= 2:
        lines.append("## Confidence Calibration\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        corr = calibration.get("confidence_cpl_correlation")
        lines.append(f"| Confidence-CPL correlation | {corr:.3f} |" if corr is not None
                     else "| Confidence-CPL correlation | N/A |")
        conf_bl = calibration.get("mean_confidence_on_blunders")
        conf_cl = calibration.get("mean_confidence_on_clean_moves")
        lines.append(f"| Mean confidence on blunders | {conf_bl:.2f} |" if conf_bl is not None
                     else "| Mean confidence on blunders | N/A |")
        lines.append(f"| Mean confidence on clean moves | {conf_cl:.2f} |" if conf_cl is not None
                     else "| Mean confidence on clean moves | N/A |")
        lines.append("")
        bins = calibration.get("bins", [])
        if bins:
            lines.append("| Confidence bin | N | Mean CPL | Blunder rate |")
            lines.append("|----------------|---|----------|--------------|")
            for b in bins:
                mean_cpl = f"{b['mean_cpl']:.1f}" if b["mean_cpl"] is not None else "—"
                br = f"{b['blunder_rate']:.1%}" if b["blunder_rate"] is not None else "—"
                lines.append(f"| {b['confidence_range']} | {b['n']} | {mean_cpl} | {br} |")
            lines.append("")

    rank_dist = run_metrics.get("candidate_rank_distribution", {})
    if rank_dist:
        lines.append("## Candidate Rank Distribution\n")
        lines.append("| Rank | Count |")
        lines.append("|------|-------|")
        for rank in sorted(rank_dist):
            lines.append(f"| {rank} | {rank_dist[rank]} |")
        lines.append("")

    return "\n".join(lines)

"""
Compare aggregate metrics across two or more experiment runs.

Usage:
    python -m src.experiments.compare_runs run_id_1 run_id_2 [run_id_3 ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run_report(run_id: str, base_dir: str = "data/runs") -> dict:
    path = Path(base_dir) / run_id / "run_report.json"
    if not path.exists():
        raise FileNotFoundError(f"No run_report.json found for run_id={run_id!r} at {path}")
    with open(path) as f:
        return json.load(f)


def load_manifest(run_id: str, base_dir: str = "data/runs") -> dict:
    path = Path(base_dir) / run_id / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def compare_runs(run_ids: list[str], base_dir: str = "data/runs") -> str:
    reports = {}
    manifests = {}
    for rid in run_ids:
        try:
            reports[rid] = load_run_report(rid, base_dir)
            manifests[rid] = load_manifest(rid, base_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            reports[rid] = {}
            manifests[rid] = {}

    METRICS = [
        ("total_games",           "Games"),
        ("win_rate",              "Win rate"),
        ("draw_rate",             "Draw rate"),
        ("loss_rate",             "Loss rate"),
        ("average_centipawn_loss","Avg CPL"),
        ("blunder_count_total",   "Total blunders"),
        ("total_token_usage",     "Total tokens"),
    ]

    MANIFEST_FIELDS = [
        ("player_type",    "Player type"),
        ("narrative_mode", "Narrative mode"),
        ("candidate_mode", "Candidate mode"),
        ("policy_name",    "Policy"),
        ("opponent_skill", "Opp skill"),
        ("prompt_version", "Prompt version"),
    ]

    col_w = max(20, max(len(rid[:30]) for rid in run_ids) + 2)
    label_w = 22

    def _row(label: str, values: list[str]) -> str:
        return f"  {label:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in values)

    lines: list[str] = []
    lines.append("\n" + "=" * (label_w + col_w * len(run_ids) + 4))
    lines.append("  EXPERIMENT COMPARISON")
    lines.append("=" * (label_w + col_w * len(run_ids) + 4))

    # Run IDs header
    lines.append(_row("Run ID", [rid[:col_w - 2] for rid in run_ids]))
    lines.append("  " + "-" * (label_w + col_w * len(run_ids)))

    lines.append("  Config")
    for field, label in MANIFEST_FIELDS:
        vals = [_fmt(manifests[rid].get(field)) for rid in run_ids]
        lines.append(_row(f"  {label}", vals))

    lines.append("")
    lines.append("  Metrics")
    for field, label in METRICS:
        vals = [_fmt(reports[rid].get(field)) for rid in run_ids]
        lines.append(_row(f"  {label}", vals))

    lines.append("")
    lines.append("  Candidate rank distribution (rank → count)")
    for rid in run_ids:
        dist = reports[rid].get("candidate_rank_distribution", {})
        lines.append(f"    {rid[:20]}: " + ", ".join(f"rank{k}={v}" for k, v in sorted(dist.items())))

    lines.append("")
    lines.append("  Presentation index distribution (positional bias check)")
    for rid in run_ids:
        dist = reports[rid].get("presentation_index_distribution", {})
        lines.append(f"    {rid[:20]}: " + ", ".join(f"idx{k}={v}" for k, v in sorted(dist.items())))

    lines.append("=" * (label_w + col_w * len(run_ids) + 4))
    output = "\n".join(lines)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment runs side by side")
    parser.add_argument("run_ids", nargs="+", help="One or more run_ids to compare")
    parser.add_argument("--dir", default="data/runs", help="Base directory for runs")
    args = parser.parse_args()
    print(compare_runs(args.run_ids, base_dir=args.dir))


if __name__ == "__main__":
    main()

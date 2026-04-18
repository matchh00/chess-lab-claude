"""
CLI trace viewer for saved game traces.

Usage:
    python -m src.ui.trace_viewer --game_id <id> [--run_id <run_id>] [--dir data/runs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import chess


def _find_game(game_id: str, base_dir: str = "data/runs") -> Optional[tuple[str, Path, Path]]:
    """Search for game_id across all runs. Returns (run_id, game_path, traces_path)."""
    base = Path(base_dir)
    if not base.exists():
        return None
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        game_path = run_dir / "games" / f"{game_id}.json"
        traces_path = run_dir / "traces" / f"{game_id}.json"
        if game_path.exists():
            return run_dir.name, game_path, traces_path
    return None


def _load_game_trace(game_path: Path) -> dict:
    with open(game_path) as f:
        return json.load(f)


def _load_move_traces(traces_path: Path) -> list[dict]:
    if not traces_path.exists():
        return []
    with open(traces_path) as f:
        return json.load(f)


def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _header(text: str, width: int = 72) -> str:
    return f"\n{'─' * 3} {text} {'─' * max(0, width - len(text) - 5)}"


def _fmt_eval(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f} cp"


def _ply_to_move_notation(ply: int) -> str:
    move_num = ply // 2 + 1
    color = "w" if ply % 2 == 0 else "b"
    return f"{move_num}{color}"


def view_game(game_id: str, run_id: Optional[str] = None, base_dir: str = "data/runs") -> None:
    if run_id:
        run_dir = Path(base_dir) / run_id
        game_path = run_dir / "games" / f"{game_id}.json"
        traces_path = run_dir / "traces" / f"{game_id}.json"
        if not game_path.exists():
            print(f"Game not found: {game_path}")
            return
    else:
        result = _find_game(game_id, base_dir)
        if result is None:
            print(f"Game {game_id!r} not found in {base_dir}/")
            return
        run_id, game_path, traces_path = result

    game = _load_game_trace(game_path)
    move_traces_raw = _load_move_traces(traces_path)

    # Index move traces by ply_index
    traces_by_ply: dict[int, dict] = {mt["ply_index"]: mt for mt in move_traces_raw}

    print(_sep("═"))
    print(f"  GAME TRACE VIEWER")
    print(f"  Game ID  : {game_id}")
    print(f"  Run ID   : {run_id}")
    print(f"  White    : {game.get('white_player', '?')}")
    print(f"  Black    : {game.get('black_player', '?')}")
    print(f"  Result   : {game.get('result', '?')}  ({game.get('termination', '?')})")
    print(f"  Plies    : {game.get('total_plies', '?')}")
    print(_sep("═"))

    for move_rec in game.get("moves", []):
        ply = move_rec["ply_index"]
        san = move_rec.get("san", "?")
        player = move_rec.get("player", "?")
        notation = _ply_to_move_notation(ply)

        print(_header(f"Ply {ply} ({notation}) — {san}  [{player}]"))
        print(f"  FEN: {move_rec.get('fen_before', '?')}")
        print(f"  Eval before: {_fmt_eval(move_rec.get('engine_eval_before'))}"
              f"  →  after: {_fmt_eval(move_rec.get('engine_eval_after'))}"
              f"  |  CPL: {move_rec.get('centipawn_loss', 'N/A')}")

        mt = traces_by_ply.get(ply)
        if mt is None:
            print(f"  [No MoveTrace — opponent move]")
            continue

        # Top 5 primitives
        ws = mt.get("weighted_state", [])
        if ws:
            top5 = sorted(ws, key=lambda s: s["weighted_score"], reverse=True)[:5]
            print(f"\n  Top 5 Primitives (weighted_score):")
            for s in top5:
                label = s.get("importance_label", "?")
                print(f"    [{label.upper():<10}] {s['primitive_id']:<40} {s['weighted_score']:.3f}")

        # Position narrative
        narrative = mt.get("position_narrative", "")
        if narrative:
            print(f"\n  Position Narrative:")
            words = narrative.split()
            line = "    "
            for word in words:
                if len(line) + len(word) > 74:
                    print(line)
                    line = "    " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)
        else:
            print(f"\n  Position Narrative: (none — narrative mode off or llm_raw)")

        # Candidates
        candidates = mt.get("candidates", [])
        if candidates:
            sorted_cands = sorted(candidates, key=lambda c: c["presentation_index"])
            print(f"\n  Candidates (in presentation order):")
            print(f"    {'[i]':<5} {'SAN':<8} {'Source':<10} {'Rank':<6} {'Eval':<12} {'Flags':<20} Narrative")
            print(f"    {'-'*5} {'-'*8} {'-'*10} {'-'*6} {'-'*12} {'-'*20} {'-'*20}")
            for c in sorted_cands:
                flags = ",".join(c.get("risk_flags", [])) or "—"
                eval_str = _fmt_eval(c.get("engine_eval_after"))
                narr = (c.get("candidate_narrative", "") or "")[:40]
                star = "★" if c.get("uci") == mt.get("decision_record", {}).get("selected_uci") else " "
                print(f"  {star} [{c['presentation_index']}]  {c['san']:<8} {c['source']:<10} "
                      f"#{c['internal_rank']:<4} {eval_str:<12} {flags:<20} {narr}")

        # Decision
        dr = mt.get("decision_record")
        if dr:
            print(f"\n  LLM Decision:")
            print(f"    Selected UCI       : {dr.get('selected_uci')}")
            print(f"    Internal rank      : {dr.get('selected_internal_rank')}")
            print(f"    Presentation index : {dr.get('selected_presentation_index')}")
            print(f"    Confidence         : {dr.get('confidence', 0.0):.2f}")
            print(f"    Fallback used      : {dr.get('fallback_used', False)}")
            reasoning = dr.get("reasoning_summary", "")
            if reasoning:
                print(f"    Reasoning          : {reasoning[:120]}")

        # Prompt info
        pr = mt.get("prompt_record")
        if pr:
            print(f"\n  Prompt: version={pr.get('prompt_version')}  tokens={pr.get('token_count')}  model={pr.get('model')}")

        # Blunder label
        blunder = mt.get("blunder_label")
        cpl = mt.get("centipawn_loss")
        if blunder:
            print(f"\n  ⚠ {blunder.upper()}: CPL={cpl:.1f}")

    print(_sep("═"))
    print(f"  End of game {game_id[:16]}...")
    print(_sep("═"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI trace viewer for saved chess games")
    parser.add_argument("--game_id", required=True, help="Game ID to view")
    parser.add_argument("--run_id", default=None, help="Run ID (optional; searched if omitted)")
    parser.add_argument("--dir", default="data/runs", help="Base directory for runs")
    args = parser.parse_args()
    view_game(args.game_id, run_id=args.run_id, base_dir=args.dir)


if __name__ == "__main__":
    main()

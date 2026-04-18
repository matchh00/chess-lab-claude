"""
Run a batch experiment from a YAML config.

Usage:
    python -m src.experiments.run_experiment configs/experiments/baseline.yaml [--games N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import chess
import yaml

from src.analytics.move_metrics import compute_blunder_label
from src.analytics.report_builder import build_reports
from src.candidates.annotator import annotate_candidates
from src.candidates.filters import filter_candidates
from src.candidates.generator import generate_engine_assisted, generate_heuristic_only
from src.candidates.ranker import rank_candidates
from src.candidates.shuffler import shuffle_candidates
from src.environment.engine_wrapper import EngineWrapper
from src.environment.opponents import make_opponent
from src.gameplay.players import LLMRawPlayer
from src.llm.client import call_llm
from src.llm.decision import choose_move
from src.llm.prompts import build_prompt
from src.narratives.candidate_narrative import build_candidate_narrative
from src.narratives.position_narrative import build_position_narrative
from src.policies.profiles import load_policy
from src.policies.summaries import text_priority_summary
from src.policies.weighting import apply_policy
from src.primitives.extractor import PrimitiveExtractor, build_default_registry
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionRecord,
    ExperimentManifest,
    GameTrace,
    MoveRecord,
    MoveTrace,
)

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _board_snapshot(board: chess.Board, game_id: str, ply: int) -> BoardSnapshot:
    return BoardSnapshot(
        game_id=game_id,
        ply_index=ply,
        fen=board.fen(),
        turn="white" if board.turn == chess.WHITE else "black",
        legal_moves=[m.uci() for m in board.legal_moves],
        move_history=[board.move_stack[i].uci() for i in range(len(board.move_stack))],
        is_check=board.is_check(),
        is_game_over=board.is_game_over(),
        result_if_terminal=board.result() if board.is_game_over() else None,
    )


def _cp_loss(eval_before: Optional[float], eval_after: Optional[float]) -> Optional[float]:
    if eval_before is None or eval_after is None:
        return None
    return max(0.0, eval_before - (-eval_after))


# ── per-ply lab move functions ────────────────────────────────────────────────

def _lab_move_llm(
    board: chess.Board,
    lab_color: chess.Color,
    engine: EngineWrapper,
    extractor: PrimitiveExtractor,
    policy,
    narrative_mode: str,
    candidate_mode: str,
    rng: random.Random,
    game_id: str,
    ply: int,
) -> tuple[str, MoveTrace]:
    snap = _board_snapshot(board, game_id, ply)
    eval_before = engine.evaluate(board)

    primitives = extractor.extract(board, lab_color)
    baseline_states = apply_policy(primitives, policy)

    # Position narrative
    if narrative_mode == "on":
        pos_budget = build_position_narrative(baseline_states, primitives, policy)
        pos_narrative = pos_budget.text
        pos_word_count = pos_budget.word_count
        pos_token_count = pos_budget.token_count
    else:
        pos_narrative = ""
        pos_word_count = 0
        pos_token_count = 0

    policy_summary = text_priority_summary(policy, baseline_states)

    # Candidates
    eval_before_for_gen = eval_before
    if candidate_mode == "engine_assisted":
        candidates = generate_engine_assisted(
            board, engine, engine_eval_before=eval_before_for_gen, rng=rng
        )
    else:
        candidates = generate_heuristic_only(board, rng=rng)

    candidates = filter_candidates(candidates, board)
    candidates = rank_candidates(candidates, board)
    candidates = shuffle_candidates(candidates, rng=rng)
    candidates = annotate_candidates(candidates, board, lab_color, extractor, policy, baseline_states)

    # Candidate narratives
    updated: list[CandidateMoveRecord] = []
    for c in candidates:
        narr = build_candidate_narrative(c)
        updated.append(c.model_copy(update={"candidate_narrative": narr.text}))
    candidates = updated

    # Prompt + decision
    prompt_record = build_prompt(snap, pos_narrative, policy_summary, candidates)
    decision = choose_move(prompt_record, candidates, board, llm_call_fn=call_llm)

    # Apply move
    move_obj = chess.Move.from_uci(decision.selected_uci)
    board.push(move_obj)
    eval_after = engine.evaluate(board)

    cp = _cp_loss(eval_before, eval_after)
    blunder = compute_blunder_label(cp)

    trace = MoveTrace(
        game_id=game_id,
        ply_index=ply,
        board_snapshot=snap,
        primitives=primitives,
        weighted_state=baseline_states,
        candidates=candidates,
        prompt_record=prompt_record,
        decision_record=decision,
        engine_eval_before=eval_before,
        engine_eval_after=eval_after,
        centipawn_loss=cp,
        blunder_label=blunder,
        token_count=prompt_record.token_count,
        position_narrative=pos_narrative,
        position_narrative_word_count=pos_word_count,
        position_narrative_token_count=pos_token_count,
    )
    return decision.selected_uci, trace


def _lab_move_llm_raw(
    board: chess.Board,
    engine: EngineWrapper,
    player: LLMRawPlayer,
    game_id: str,
    ply: int,
) -> tuple[str, MoveTrace]:
    snap = _board_snapshot(board, game_id, ply)
    eval_before = engine.evaluate(board)

    decision = player.decide(board)

    move_obj = chess.Move.from_uci(decision.selected_uci)
    board.push(move_obj)
    eval_after = engine.evaluate(board)

    cp = _cp_loss(eval_before, eval_after)
    blunder = compute_blunder_label(cp)

    trace = MoveTrace(
        game_id=game_id,
        ply_index=ply,
        board_snapshot=snap,
        primitives=[],
        weighted_state=[],
        candidates=[],
        prompt_record=None,
        decision_record=decision,
        engine_eval_before=eval_before,
        engine_eval_after=eval_after,
        centipawn_loss=cp,
        blunder_label=blunder,
        token_count=0,
    )
    return decision.selected_uci, trace


# ── single game ───────────────────────────────────────────────────────────────

def play_game(
    config: dict,
    engine: EngineWrapper,
    extractor: Optional[PrimitiveExtractor],
    policy,
    llm_raw_player: Optional[LLMRawPlayer],
    rng: random.Random,
    game_id: str,
    game_num: int,
) -> tuple[GameTrace, list[MoveTrace]]:
    player_type = config["player_type"]
    narrative_mode = config.get("narrative_mode", "on")
    candidate_mode = config.get("candidate_mode", "engine_assisted")
    max_moves = config.get("max_moves_per_game", 30)
    lab_color = chess.WHITE if config.get("lab_color", "white") == "white" else chess.BLACK

    opp_cfg = config["opponent"]
    opponent = make_opponent(
        opp_cfg["type"],
        engine_path=opp_cfg.get("engine_path", "/opt/homebrew/bin/stockfish"),
        skill_level=opp_cfg.get("skill_level", 3),
        depth=opp_cfg.get("depth", 8),
        time_limit=opp_cfg.get("time_limit", 0.1),
    )
    opponent.on_game_start()

    board = chess.Board()
    move_records: list[MoveRecord] = []
    move_traces: list[MoveTrace] = []
    white_name = "lab" if lab_color == chess.WHITE else opp_cfg["type"]
    black_name = "lab" if lab_color == chess.BLACK else opp_cfg["type"]

    logger.info("Game %d/%s starting (player=%s, narrative=%s)",
                game_num, game_id[:8], player_type, narrative_mode)

    try:
        for ply in range(max_moves * 2):
            if board.is_game_over():
                break

            fen_before = board.fen()
            is_lab_turn = board.turn == lab_color

            if is_lab_turn:
                if player_type == "llm":
                    uci, mt = _lab_move_llm(
                        board, lab_color, engine, extractor, policy,
                        narrative_mode, candidate_mode, rng, game_id, ply
                    )
                    move_traces.append(mt)
                    eval_before = mt.engine_eval_before
                    eval_after = mt.engine_eval_after
                elif player_type == "llm_raw":
                    uci, mt = _lab_move_llm_raw(board, engine, llm_raw_player, game_id, ply)
                    # Note: board already pushed inside _lab_move_llm_raw
                    move_traces.append(mt)
                    eval_before = mt.engine_eval_before
                    eval_after = mt.engine_eval_after
                elif player_type == "random":
                    eval_before = engine.evaluate(board)
                    move = rng.choice(list(board.legal_moves))
                    uci = move.uci()
                    board.push(move)
                    eval_after = engine.evaluate(board)
                elif player_type == "stockfish":
                    eval_before = engine.evaluate(board)
                    moves = engine.best_move(board, top_n=1)
                    uci = moves[0] if moves else list(board.legal_moves)[0].uci()
                    board.push(chess.Move.from_uci(uci))
                    eval_after = engine.evaluate(board)
                else:
                    raise ValueError(f"Unknown player_type: {player_type!r}")

                # For non-LLM players, compute SAN from fen_before
                temp_board = chess.Board(fen_before)
                san = temp_board.san(chess.Move.from_uci(uci))
                cp = _cp_loss(eval_before, eval_after)

                move_records.append(MoveRecord(
                    ply_index=ply, uci=uci, san=san,
                    fen_before=fen_before, fen_after=board.fen(),
                    player="self",
                    engine_eval_before=eval_before,
                    engine_eval_after=eval_after,
                    centipawn_loss=cp,
                ))
            else:
                uci = opponent.choose_move(board)
                temp_board = chess.Board(fen_before)
                san = temp_board.san(chess.Move.from_uci(uci))
                board.push(chess.Move.from_uci(uci))
                move_records.append(MoveRecord(
                    ply_index=ply, uci=uci, san=san,
                    fen_before=fen_before, fen_after=board.fen(),
                    player="opponent",
                ))

    finally:
        opponent.on_game_end()

    result = board.result() if board.is_game_over() else None
    game_trace = GameTrace(
        game_id=game_id,
        end_time=datetime.now(UTC),
        config=config,
        white_player=white_name,
        black_player=black_name,
        moves=move_records,
        result=result,
        termination=_termination(board),
        total_plies=len(move_records),
    )

    white_losses = [m.centipawn_loss for m in move_records
                    if m.centipawn_loss is not None and m.ply_index % 2 == 0]
    black_losses = [m.centipawn_loss for m in move_records
                    if m.centipawn_loss is not None and m.ply_index % 2 == 1]
    if white_losses:
        game_trace.average_centipawn_loss_white = sum(white_losses) / len(white_losses)
    if black_losses:
        game_trace.average_centipawn_loss_black = sum(black_losses) / len(black_losses)

    logger.info("Game %s finished: %s (%s), %d plies, %d lab moves",
                game_id[:8], result, game_trace.termination, len(move_records), len(move_traces))
    return game_trace, move_traces


def _termination(board: chess.Board) -> Optional[str]:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves():
        return "seventy_five_moves"
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    return None


# ── experiment runner ─────────────────────────────────────────────────────────

def run_experiment(config_path: str, game_count_override: Optional[int] = None,
                   seed_override: Optional[int] = None) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if game_count_override is not None:
        config["game_count"] = game_count_override
    if seed_override is not None:
        config["random_seed"] = seed_override

    log_level = getattr(logging, config.get("logging_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_id = config.get("config_id", Path(config_path).stem)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{config_id}_{ts}"
    run_dir = Path("data") / "runs" / run_id
    (run_dir / "games").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (Path("reports") / "latest").mkdir(parents=True, exist_ok=True)

    seed = config.get("random_seed")
    rng = random.Random(seed)

    player_type = config["player_type"]
    policy_name = config.get("policy")
    opp_cfg = config["opponent"]
    eng_cfg = config.get("engine", {})

    # Shared engine for candidate generation + eval
    engine = EngineWrapper(
        path=eng_cfg.get("path", "/opt/homebrew/bin/stockfish"),
        skill_level=eng_cfg.get("skill_level", 5),
        depth=eng_cfg.get("depth", 10),
        time_limit=eng_cfg.get("time_limit", 0.1),
    )
    engine.open()

    extractor = None
    policy = None
    llm_raw_player = None

    if player_type == "llm":
        extractor = PrimitiveExtractor(build_default_registry())
        policy = load_policy(policy_name)
    elif player_type == "llm_raw":
        llm_raw_player = LLMRawPlayer(llm_call_fn=call_llm)

    manifest = ExperimentManifest(
        run_id=run_id,
        config_path=config_path,
        config=config,
        total_games=config.get("game_count", 20),
        player_type=player_type,
        narrative_mode=config.get("narrative_mode", "on"),
        candidate_mode=config.get("candidate_mode", "engine_assisted") or "",
        policy_name=policy_name or "",
        opponent_type=opp_cfg.get("type", "stockfish"),
        opponent_skill=opp_cfg.get("skill_level", 3),
        prompt_version=config.get("prompt_version", "v1.1"),
        random_seed=seed,
    )

    all_game_traces: list[GameTrace] = []
    all_move_traces: list[MoveTrace] = []
    traces_by_game: dict[str, list[MoveTrace]] = {}

    game_count = config.get("game_count", 20)
    print(f"\nRun: {run_id}")
    print(f"Config: {config_path}  |  {game_count} games  |  player={player_type}  |  narrative={manifest.narrative_mode}\n")

    for i in range(game_count):
        game_id = str(uuid.uuid4())
        print(f"  Game {i+1}/{game_count}  {game_id[:8]}...", end="", flush=True)
        try:
            gt, mts = play_game(
                config=config,
                engine=engine,
                extractor=extractor,
                policy=policy,
                llm_raw_player=llm_raw_player,
                rng=rng,
                game_id=game_id,
                game_num=i + 1,
            )
            # Save game trace
            gt_path = run_dir / "games" / f"{game_id}.json"
            gt.save(str(gt_path))

            # Save move traces
            mt_path = run_dir / "traces" / f"{game_id}.json"
            with open(mt_path, "w") as f:
                f.write("[" + ",\n".join(mt.to_json() for mt in mts) + "]")

            all_game_traces.append(gt)
            all_move_traces.extend(mts)
            traces_by_game[game_id] = mts
            manifest.game_ids.append(game_id)
            manifest.completed_games += 1

            result_str = gt.result or "stopped"
            cpl = (gt.average_centipawn_loss_white
                   if gt.white_player == "lab" else gt.average_centipawn_loss_black)
            cpl_str = f"{cpl:.1f}" if cpl is not None else "N/A"
            print(f"  result={result_str}  avg_cpl={cpl_str}  lab_moves={len(mts)}")

        except Exception as e:
            manifest.failed_games += 1
            print(f"  FAILED: {e}")
            logger.exception("Game %s failed", game_id)

    engine.close()

    manifest.end_time = datetime.now(UTC)
    manifest.save(str(run_dir / "manifest.json"))
    print(f"\nManifest saved → {run_dir / 'manifest.json'}")

    # Auto-run analytics
    print("Running analytics...", flush=True)
    paths = build_reports(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        all_move_traces=all_move_traces,
        all_game_traces=all_game_traces,
        traces_by_game=traces_by_game,
    )
    print("Reports generated:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chess lab experiment")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--games", type=int, default=None, help="Override game_count")
    parser.add_argument("--seed", type=int, default=None, help="Override random_seed")
    args = parser.parse_args()

    run_id = run_experiment(args.config, game_count_override=args.games, seed_override=args.seed)
    print(f"\nDone. run_id={run_id}")


if __name__ == "__main__":
    main()

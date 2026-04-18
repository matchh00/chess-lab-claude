#!/usr/bin/env python3
"""Run 10 plies: LLMPlayer (white, balanced policy) vs Stockfish skill-3 (black).
Prints the full MoveTrace for ply 4 (move 5, 0-indexed).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from src.environment.engine_wrapper import EngineWrapper
from src.environment.opponents import make_opponent
from src.candidates.generator import generate_engine_assisted
from src.candidates.filters import filter_candidates
from src.candidates.ranker import rank_candidates
from src.candidates.shuffler import shuffle_candidates
from src.candidates.annotator import annotate_candidates
from src.narratives.position_narrative import build_position_narrative
from src.narratives.candidate_narrative import build_candidate_narrative
from src.policies.profiles import load_policy
from src.policies.weighting import apply_policy
from src.primitives.extractor import PrimitiveExtractor, build_default_registry
from src.llm.prompts import build_prompt
from src.llm.decision import choose_move
from src.llm.client import call_llm
from src.storage.models import (
    BoardSnapshot,
    MoveTrace,
    CandidateMoveRecord,
)

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
MAX_PLIES = 10
TARGET_PLY = 4  # 0-indexed — this is "move 5"


def board_to_snapshot(board: chess.Board, game_id: str, ply: int) -> BoardSnapshot:
    turn = "white" if board.turn == chess.WHITE else "black"
    return BoardSnapshot(
        game_id=game_id,
        ply_index=ply,
        fen=board.fen(),
        turn=turn,
        legal_moves=[m.uci() for m in board.legal_moves],
        move_history=[board.move_stack[i].uci() for i in range(len(board.move_stack))],
        is_check=board.is_check(),
        is_game_over=board.is_game_over(),
        result_if_terminal=board.result() if board.is_game_over() else None,
    )


def run():
    game_id = "demo-llm-vs-sf3"
    policy = load_policy("balanced")
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)

    engine = EngineWrapper(
        path=STOCKFISH_PATH,
        skill_level=5,  # engine used for candidate generation/eval
        depth=10,
        time_limit=0.1,
    )
    engine.open()

    opponent = make_opponent(
        "stockfish",
        engine_path=STOCKFISH_PATH,
        skill_level=3,
        depth=8,
        time_limit=0.1,
    )
    opponent.on_game_start()

    board = chess.Board()
    lab_color = chess.WHITE
    move_traces: dict[int, MoveTrace] = {}

    print(f"Starting game: LLMPlayer (white) vs Stockfish-skill-3 (black) — {MAX_PLIES} plies\n")

    for ply in range(MAX_PLIES):
        if board.is_game_over():
            print(f"Game over at ply {ply}: {board.result()}")
            break

        snap = board_to_snapshot(board, game_id, ply)

        if board.turn == lab_color:
            print(f"  Ply {ply} (white/LLM) — generating candidates...")

            # Eval before
            eval_before = engine.evaluate(board)

            # Baseline primitives
            primitives_before = extractor.extract(board, lab_color)
            baseline_states = apply_policy(primitives_before, policy)

            # Candidate pipeline
            candidates = generate_engine_assisted(board, engine, engine_eval_before=eval_before)
            candidates = filter_candidates(candidates, board)
            candidates = rank_candidates(candidates, board)
            candidates = shuffle_candidates(candidates)
            candidates = annotate_candidates(
                candidates, board, lab_color, extractor, policy, baseline_states
            )

            # Build candidate narratives
            updated: list[CandidateMoveRecord] = []
            for c in candidates:
                budget = build_candidate_narrative(c)
                updated.append(c.model_copy(update={"candidate_narrative": budget.text}))
            candidates = updated

            # Position narrative
            pos_narrative = build_position_narrative(baseline_states, primitives_before, policy)

            # Policy summary (simple)
            from src.policies.summaries import text_priority_summary
            policy_summary = text_priority_summary(policy, baseline_states)

            # Build prompt and call LLM
            prompt_record = build_prompt(
                snap, pos_narrative.text, policy_summary, candidates
            )
            print(f"           prompt tokens: {prompt_record.token_count}")

            decision = choose_move(prompt_record, candidates, board, llm_call_fn=call_llm)
            print(f"           LLM selected: {decision.selected_uci}  (rank={decision.selected_internal_rank})")

            # Compute SAN before pushing (board must be in pre-move state)
            move_obj = chess.Move.from_uci(decision.selected_uci)
            san = board.san(move_obj)
            board.push(move_obj)
            eval_after = engine.evaluate(board)

            cp_loss = None
            if eval_before is not None and eval_after is not None:
                cp_loss = max(0.0, eval_before - (-eval_after))

            trace = MoveTrace(
                game_id=game_id,
                ply_index=ply,
                board_snapshot=snap,
                primitives=primitives_before,
                weighted_state=baseline_states,
                candidates=candidates,
                prompt_record=prompt_record,
                decision_record=decision,
                engine_eval_before=eval_before,
                engine_eval_after=eval_after,
                centipawn_loss=cp_loss,
                token_count=prompt_record.token_count,
                position_narrative=pos_narrative.text,
                position_narrative_word_count=pos_narrative.word_count,
                position_narrative_token_count=pos_narrative.token_count,
            )
            move_traces[ply] = trace
            print(f"           san={san}  eval_before={eval_before}  eval_after={eval_after}")

        else:
            uci = opponent.choose_move(board)
            san = board.san(chess.Move.from_uci(uci))
            board.push(chess.Move.from_uci(uci))
            print(f"  Ply {ply} (black/Stockfish): {san}")

    opponent.on_game_end()
    engine.close()

    # ── Print MoveTrace for ply TARGET_PLY ──────────────────────────────────
    print("\n" + "=" * 70)
    print(f"MOVETRACE — Ply {TARGET_PLY} (move {TARGET_PLY + 1}, white)")
    print("=" * 70)

    if TARGET_PLY not in move_traces:
        print(f"No MoveTrace recorded for ply {TARGET_PLY}.")
        return

    mt = move_traces[TARGET_PLY]
    pr = mt.prompt_record
    dr = mt.decision_record

    print(f"\n── Board ────────────────────────────────────────────────────")
    print(f"  game_id               : {mt.game_id}")
    print(f"  ply_index             : {mt.ply_index}")
    print(f"  fen                   : {mt.board_snapshot.fen}")
    print(f"  turn                  : {mt.board_snapshot.turn}")
    print(f"  is_check              : {mt.board_snapshot.is_check}")

    print(f"\n── Evaluation ───────────────────────────────────────────────")
    print(f"  engine_eval_before    : {mt.engine_eval_before}")
    print(f"  engine_eval_after     : {mt.engine_eval_after}")
    print(f"  centipawn_loss        : {mt.centipawn_loss}")
    print(f"  blunder_label         : {mt.blunder_label}")

    print(f"\n── Position Narrative ───────────────────────────────────────")
    print(f"  word_count            : {mt.position_narrative_word_count}")
    print(f"  token_count           : {mt.position_narrative_token_count}")
    print(f"  text                  : {mt.position_narrative}")

    print(f"\n── Prompt ───────────────────────────────────────────────────")
    print(f"  prompt_version        : {pr.prompt_version}")
    print(f"  model                 : {pr.model}")
    print(f"  temperature           : {pr.temperature}")
    print(f"  token_count           : {pr.token_count}")

    print(f"\n── Candidates ({len(mt.candidates)}) ─────────────────────────────────────────")
    for c in sorted(mt.candidates, key=lambda x: x.presentation_index):
        print(f"  [{c.presentation_index}] {c.san:6s}  uci={c.uci}  "
              f"rank={c.internal_rank}  src={c.source:9s}  "
              f"eval={c.engine_eval_after!r:>8}  "
              f"flags={c.risk_flags or '[]'}")

    print(f"\n── Decision ─────────────────────────────────────────────────")
    print(f"  selected_uci          : {dr.selected_uci}")
    print(f"  selected_internal_rank: {dr.selected_internal_rank}")
    print(f"  selected_presentation : {dr.selected_presentation_index}")
    print(f"  confidence            : {dr.confidence}")
    print(f"  fallback_used         : {dr.fallback_used}")
    print(f"  is_valid              : {dr.is_valid}")
    print(f"  reasoning_summary     : {dr.reasoning_summary}")

    print(f"\n── Move Trace Totals ─────────────────────────────────────────")
    print(f"  total candidates      : {len(mt.candidates)}")
    print(f"  total primitives      : {len(mt.primitives)}")
    print(f"  weighted_state count  : {len(mt.weighted_state)}")
    print(f"  token_count (trace)   : {mt.token_count}")


if __name__ == "__main__":
    run()

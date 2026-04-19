"""
Learning runner — wraps the existing play_game loop with a cybernetic feedback hook.

Usage:
    python -m src.learning.learning_runner configs/experiments/learning_baseline.yaml
    python -m src.learning.learning_runner configs/experiments/learning_baseline.yaml --games 5
"""
from __future__ import annotations

import argparse
import copy
import logging
import random
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import yaml

from src.analytics.report_builder import build_reports
from src.environment.engine_wrapper import EngineWrapper
from src.experiments.run_experiment import play_game
from src.gameplay.players import LLMRawPlayer
from src.storage.models import ExperimentManifest, GameTrace, MoveTrace
from src.learning.evaluator import evaluate_move
from src.learning.influence import score_move_influence, InfluenceRecord
from src.learning.learning_log import GameAdjustmentEntry, LearningLog
from src.learning.rebalancer import rebalance_weights, save_learned_policy, WeightAdjustment
from src.learning.summary import (
    LearningGameSummary,
    PerformanceRow,
    build_game_summary,
    build_primitive_summaries,
)
from src.llm.client import call_llm
from src.policies.profiles import load_policy
from src.primitives.extractor import PrimitiveExtractor, build_default_registry

logger = logging.getLogger(__name__)


def run_with_learning(
    config_path: str,
    game_count_override: Optional[int] = None,
    seed_override: Optional[int] = None,
) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if game_count_override is not None:
        config["game_count"] = game_count_override
    if seed_override is not None:
        config["random_seed"] = seed_override

    learning_mode: bool = config.get("learning_mode", False)
    gain: float = float(config.get("gain", 0.05))
    good_threshold: float = float(config.get("good_threshold", 30.0))
    bad_threshold: float = float(config.get("bad_threshold", 100.0))

    log_level = getattr(logging, config.get("logging_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config_id = config.get("config_id", Path(config_path).stem)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{config_id}_{ts}"
    run_dir = Path("data") / "runs" / run_id
    (run_dir / "games").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    seed = config.get("random_seed")
    rng = random.Random(seed)

    player_type = config["player_type"]
    policy_name = config.get("policy", "balanced")
    eng_cfg = config.get("engine", {})

    engine = EngineWrapper(
        path=eng_cfg.get("path", "/opt/homebrew/bin/stockfish"),
        skill_level=eng_cfg.get("skill_level", 5),
        depth=eng_cfg.get("depth", 10),
        time_limit=eng_cfg.get("time_limit", 0.1),
    )
    engine.open()

    extractor: Optional[PrimitiveExtractor] = None
    policy = None
    llm_raw_player: Optional[LLMRawPlayer] = None

    if player_type == "llm":
        extractor = PrimitiveExtractor(build_default_registry())
        policy = load_policy(policy_name)
    elif player_type == "llm_raw":
        llm_raw_player = LLMRawPlayer(llm_call_fn=call_llm)

    original_weights: dict[str, float] = copy.deepcopy(policy.weight_map) if policy else {}

    game_count = config.get("game_count", 20)
    print(f"\nRun: {run_id}  |  learning_mode={learning_mode}")
    print(f"Config: {config_path}  |  {game_count} games  |  player={player_type}\n")

    opp_cfg = config["opponent"]
    manifest = ExperimentManifest(
        run_id=run_id,
        config_path=config_path,
        config=config,
        total_games=game_count,
        player_type=player_type,
        narrative_mode=config.get("narrative_mode", "on"),
        candidate_mode=config.get("candidate_mode", "engine_assisted") or "",
        policy_name=policy_name or "",
        opponent_type=opp_cfg.get("type", "stockfish"),
        opponent_skill=opp_cfg.get("skill_level", 3),
        prompt_version=config.get("prompt_version", "v1.1"),
        random_seed=seed,
    )

    learning_log = LearningLog(run_id=run_id)
    summary = LearningGameSummary(run_id=run_id)
    all_adjustments: list[WeightAdjustment] = []
    all_influence: list[InfluenceRecord] = []
    all_game_traces: list[GameTrace] = []
    all_move_traces: list[MoveTrace] = []
    traces_by_game: dict[str, list[MoveTrace]] = {}

    try:
        for i in range(game_count):
            game_id = str(uuid.uuid4())
            print(f"  Game {i+1}/{game_count}  {game_id[:8]}...", end="", flush=True)

            game_trace, move_traces = play_game(
                config=config,
                engine=engine,
                extractor=extractor,
                policy=policy,
                llm_raw_player=llm_raw_player,
                rng=rng,
                game_id=game_id,
                game_num=i + 1,
            )

            game_trace.save(str(run_dir / "games" / f"{game_id}.json"))
            mt_path = run_dir / "traces" / f"{game_id}.json"
            with open(mt_path, "w") as f:
                f.write("[" + ",\n".join(mt.to_json() for mt in move_traces) + "]")

            all_game_traces.append(game_trace)
            all_move_traces.extend(move_traces)
            traces_by_game[game_id] = move_traces
            manifest.game_ids.append(game_id)
            manifest.completed_games += 1

            print(f"  result={game_trace.result or 'limit'}  plies={game_trace.total_plies}")

            evaluations = [evaluate_move(t) for t in move_traces]
            influence_records = [
                score_move_influence(t, ev, good_threshold=good_threshold, bad_threshold=bad_threshold)
                for t, ev in zip(move_traces, evaluations)
                if t.weighted_state
            ]
            all_influence.extend(influence_records)

            good_count = sum(1 for r in influence_records if r.is_good)
            bad_count = sum(1 for r in influence_records if r.is_bad)

            row = build_game_summary(run_id, i + 1, game_id, move_traces, influence_records)
            summary.performance_rows.append(row)

            adjustments: list[WeightAdjustment] = []
            if learning_mode and policy and influence_records:
                adjustments = rebalance_weights(
                    influence_records=influence_records,
                    policy=policy,
                    original_weights=original_weights,
                    gain=gain,
                )
                all_adjustments.extend(adjustments)
                logger.info("Game %d: %d adjustments applied", i + 1, len(adjustments))

            learning_log.add(GameAdjustmentEntry(
                game_num=i + 1,
                game_id=game_id,
                ply_count=game_trace.total_plies,
                good_move_count=good_count,
                bad_move_count=bad_count,
                adjustments=adjustments,
            ))

    finally:
        engine.close()

    manifest.end_time = datetime.now(UTC)
    manifest.save(str(run_dir / "manifest.json"))

    print("Running analytics...", flush=True)
    paths = build_reports(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        all_move_traces=all_move_traces,
        all_game_traces=all_game_traces,
        traces_by_game=traces_by_game,
    )
    for name, path in paths.items():
        print(f"  {name}: {path}")

    learning_log.save(run_dir / "learning_log.json")

    if learning_mode and policy:
        learned_path = save_learned_policy(policy, policy_name, run_id)
        logger.info("Learned policy saved to %s", learned_path)
        final_weights = copy.deepcopy(policy.weight_map)
        summary.primitive_summaries = build_primitive_summaries(
            original_weights, final_weights, all_adjustments
        )

    print(f"\nLearning run complete: {run_id}")
    print(f"Log: {run_dir / 'learning_log.json'}")
    return run_id


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run a learning experiment")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--games", type=int, help="Override game_count from config")
    parser.add_argument("--seed", type=int, help="Override random_seed from config")
    args = parser.parse_args()
    run_with_learning(args.config, game_count_override=args.games, seed_override=args.seed)


if __name__ == "__main__":
    _main()

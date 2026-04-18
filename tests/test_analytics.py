from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.move_metrics import (
    BLUNDER_THRESHOLD,
    INACCURACY_THRESHOLD,
    MISTAKE_THRESHOLD,
    compute_blunder_label,
    extract_move_row,
    build_move_log_df,
)
from src.analytics.game_metrics import accuracy_estimate, compute_game_metrics, build_game_summary_df
from src.analytics.run_metrics import compute_run_metrics
from src.analytics.primitive_attribution import compute_primitive_attribution
from src.analytics.report_builder import build_reports
from src.storage.models import (
    BoardSnapshot,
    CandidateMoveRecord,
    DecisionPromptRecord,
    DecisionRecord,
    ExperimentManifest,
    GameTrace,
    MoveRecord,
    MoveTrace,
    PrimitiveValue,
    WeightedPrimitiveState,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _snapshot(game_id: str = "g1", ply: int = 0) -> BoardSnapshot:
    return BoardSnapshot(
        game_id=game_id, ply_index=ply,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        turn="white", legal_moves=["e2e4"], move_history=[],
        is_check=False, is_game_over=False,
    )


def _decision(uci: str = "e2e4", rank: int = 1, pidx: int = 0,
               confidence: float = 0.8) -> DecisionRecord:
    return DecisionRecord(
        selected_uci=uci, selected_internal_rank=rank,
        selected_presentation_index=pidx, confidence=confidence,
    )


def _prompt_record(tokens: int = 500) -> DecisionPromptRecord:
    return DecisionPromptRecord(
        system_prompt="sys", user_prompt="usr", token_count=tokens
    )


def _weighted_state(primitive_id: str, category: str,
                    weighted_score: float, normalized_value: float = 0.7) -> WeightedPrimitiveState:
    return WeightedPrimitiveState(
        primitive_id=primitive_id, name=primitive_id, category=category,
        raw_value=1, normalized_value=normalized_value,
        weight=1.0, confidence=1.0, weighted_score=weighted_score,
        importance_label="medium", policy_message="",
    )


def _move_trace(
    game_id: str = "g1", ply: int = 0,
    centipawn_loss: float | None = None,
    blunder_label: str | None = None,
    rank: int = 1, pidx: int = 0,
    confidence: float = 0.8,
    token_count: int = 500,
    weighted_scores: dict[str, float] | None = None,
) -> MoveTrace:
    ws = [
        _weighted_state(pid, "material", score)
        for pid, score in (weighted_scores or {"self_material_difference": 0.8}).items()
    ]
    return MoveTrace(
        game_id=game_id, ply_index=ply,
        board_snapshot=_snapshot(game_id, ply),
        primitives=[], weighted_state=ws, candidates=[],
        prompt_record=_prompt_record(token_count),
        decision_record=_decision("e2e4", rank, pidx, confidence),
        engine_eval_before=50.0, engine_eval_after=-30.0,
        centipawn_loss=centipawn_loss, blunder_label=blunder_label,
        token_count=token_count,
    )


def _game_trace(game_id: str = "g1", result: str = "1-0") -> GameTrace:
    return GameTrace(
        game_id=game_id, white_player="lab", black_player="stockfish",
        result=result,
        moves=[
            MoveRecord(ply_index=0, uci="e2e4", san="e4",
                       fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                       fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                       player="self", centipawn_loss=20.0),
        ],
    )


def _manifest(run_id: str = "test_run") -> ExperimentManifest:
    return ExperimentManifest(
        run_id=run_id, total_games=5, completed_games=5,
        player_type="llm", narrative_mode="on",
        candidate_mode="engine_assisted", policy_name="balanced",
    )


# ── blunder label thresholds ──────────────────────────────────────────────────

class TestComputeBlunderLabel:
    def test_blunder(self):
        assert compute_blunder_label(BLUNDER_THRESHOLD + 1) == "blunder"

    def test_blunder_exact(self):
        assert compute_blunder_label(201.0) == "blunder"

    def test_mistake(self):
        assert compute_blunder_label(150.0) == "mistake"

    def test_mistake_boundary(self):
        assert compute_blunder_label(MISTAKE_THRESHOLD + 1) == "mistake"

    def test_inaccuracy(self):
        assert compute_blunder_label(75.0) == "inaccuracy"

    def test_inaccuracy_boundary(self):
        assert compute_blunder_label(INACCURACY_THRESHOLD + 1) == "inaccuracy"

    def test_clean(self):
        assert compute_blunder_label(30.0) is None

    def test_zero(self):
        assert compute_blunder_label(0.0) is None

    def test_none_input(self):
        assert compute_blunder_label(None) is None

    def test_exact_thresholds_are_exclusive(self):
        # >200 = blunder; exactly 200 is mistake
        assert compute_blunder_label(200.0) == "mistake"
        assert compute_blunder_label(100.0) == "inaccuracy"
        assert compute_blunder_label(50.0) is None


# ── centipawn loss calculation ────────────────────────────────────────────────

class TestCentipawnLoss:
    def _cp_loss(self, eval_before, eval_after):
        if eval_before is None or eval_after is None:
            return None
        return max(0.0, eval_before - (-eval_after))

    def test_good_move_small_loss(self):
        # Up 50cp, make good move that stays up 45cp (eval_after=-45 from opp perspective)
        cp = self._cp_loss(50.0, -45.0)
        assert abs(cp - 5.0) < 0.01

    def test_bad_move_large_loss(self):
        # Up 50cp, blunder gives opponent 200cp advantage
        cp = self._cp_loss(50.0, 200.0)
        assert cp == 250.0

    def test_equal_position_no_loss(self):
        cp = self._cp_loss(0.0, 0.0)
        assert cp == 0.0

    def test_improvement_is_zero_loss(self):
        # Position improves: we were down 50, after move opponent is down 30
        cp = self._cp_loss(-50.0, -30.0)
        assert cp == 0.0  # max(0, -50 - 30) = max(0, -80) = 0

    def test_none_eval_returns_none(self):
        assert self._cp_loss(None, 50.0) is None
        assert self._cp_loss(50.0, None) is None


# ── move_metrics ──────────────────────────────────────────────────────────────

class TestExtractMoveRow:
    def test_returns_dict(self):
        mt = _move_trace(centipawn_loss=30.0)
        row = extract_move_row(mt)
        assert isinstance(row, dict)

    def test_game_id_present(self):
        mt = _move_trace(game_id="game123", centipawn_loss=50.0)
        row = extract_move_row(mt)
        assert row["game_id"] == "game123"

    def test_centipawn_loss_present(self):
        mt = _move_trace(centipawn_loss=80.0)
        row = extract_move_row(mt)
        assert row["centipawn_loss"] == 80.0

    def test_rank_and_pidx(self):
        mt = _move_trace(rank=2, pidx=3)
        row = extract_move_row(mt)
        assert row["selected_internal_rank"] == 2
        assert row["selected_presentation_index"] == 3

    def test_token_count_from_prompt_record(self):
        mt = _move_trace(token_count=750)
        row = extract_move_row(mt)
        assert row["token_count"] == 750


class TestBuildMoveLogDf:
    def test_empty_returns_empty_df(self):
        df = build_move_log_df([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_correct_row_count(self):
        traces = [_move_trace(ply=i) for i in range(5)]
        df = build_move_log_df(traces)
        assert len(df) == 5

    def test_has_required_columns(self):
        df = build_move_log_df([_move_trace()])
        for col in ("game_id", "ply_index", "centipawn_loss", "selected_internal_rank"):
            assert col in df.columns


# ── game_metrics ──────────────────────────────────────────────────────────────

class TestAccuracyEstimate:
    def test_perfect_accuracy_near_100(self):
        acc = accuracy_estimate(0.0)
        assert acc > 95

    def test_decreases_with_higher_cpl(self):
        assert accuracy_estimate(50.0) > accuracy_estimate(100.0)

    def test_bounded_0_to_100(self):
        for cpl in [0, 50, 100, 200, 500]:
            acc = accuracy_estimate(cpl)
            assert 0.0 <= acc <= 100.0


class TestComputeGameMetrics:
    def test_returns_dict(self):
        gt = _game_trace()
        mts = [_move_trace(centipawn_loss=30.0)]
        metrics = compute_game_metrics(gt, mts)
        assert isinstance(metrics, dict)

    def test_game_id(self):
        gt = _game_trace("mygame")
        metrics = compute_game_metrics(gt, [])
        assert metrics["game_id"] == "mygame"

    def test_result(self):
        gt = _game_trace(result="0-1")
        metrics = compute_game_metrics(gt, [])
        assert metrics["result"] == "0-1"

    def test_blunder_count(self):
        mts = [
            _move_trace(centipawn_loss=250.0),  # blunder
            _move_trace(centipawn_loss=30.0),   # clean
        ]
        metrics = compute_game_metrics(_game_trace(), mts)
        assert metrics["blunder_count"] == 1

    def test_avg_cpl(self):
        mts = [
            _move_trace(centipawn_loss=100.0),
            _move_trace(centipawn_loss=200.0),
        ]
        metrics = compute_game_metrics(_game_trace(), mts)
        assert abs(metrics["avg_centipawn_loss"] - 150.0) < 0.01

    def test_avg_cpl_none_when_no_traces(self):
        metrics = compute_game_metrics(_game_trace(), [])
        assert metrics["avg_centipawn_loss"] is None


# ── primitive attribution ─────────────────────────────────────────────────────

class TestComputePrimitiveAttribution:
    def test_returns_dataframe(self):
        traces = [_move_trace(centipawn_loss=50.0)]
        df = compute_primitive_attribution(traces)
        assert isinstance(df, pd.DataFrame)

    def test_empty_when_no_traces(self):
        df = compute_primitive_attribution([])
        assert df.empty

    def test_has_required_columns(self):
        traces = [_move_trace(centipawn_loss=float(i * 20)) for i in range(10)]
        df = compute_primitive_attribution(traces)
        if not df.empty:
            assert "primitive_id" in df.columns
            assert "correlation" in df.columns
            assert "n_observations" in df.columns

    def test_runs_without_error_on_varied_data(self):
        traces = [
            _move_trace(centipawn_loss=float(i * 10),
                        weighted_scores={"self_material_difference": i * 0.1,
                                         "self_pawn_shield_quality": (10 - i) * 0.1})
            for i in range(1, 11)
        ]
        df = compute_primitive_attribution(traces)
        assert isinstance(df, pd.DataFrame)

    def test_sorted_by_abs_correlation(self):
        traces = [
            _move_trace(centipawn_loss=float(i * 20),
                        weighted_scores={"self_strong_signal": i * 0.1,
                                         "self_weak_signal": 0.5})
            for i in range(1, 15)
        ]
        df = compute_primitive_attribution(traces)
        if len(df) >= 2:
            corrs = df["correlation"].abs().tolist()
            assert corrs == sorted(corrs, reverse=True)

    def test_skips_traces_without_weighted_state(self):
        mt_with = _move_trace(centipawn_loss=50.0)
        mt_without = MoveTrace(
            game_id="g1", ply_index=1,
            board_snapshot=_snapshot(), centipawn_loss=80.0,
        )
        df = compute_primitive_attribution([mt_with, mt_without])
        assert isinstance(df, pd.DataFrame)


# ── report files created ──────────────────────────────────────────────────────

class TestBuildReports:
    def test_report_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "data" / "runs" / "test_run"
            reports_dir = Path(tmpdir) / "reports" / "latest"
            (run_dir / "games").mkdir(parents=True)
            (run_dir / "traces").mkdir(parents=True)
            reports_dir.mkdir(parents=True)

            # Temporarily change working dir so relative paths resolve correctly
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                gt = _game_trace()
                mts = [_move_trace(centipawn_loss=30.0, game_id=gt.game_id)]
                manifest = _manifest("test_run")

                paths = build_reports(
                    run_id="test_run",
                    run_dir=run_dir,
                    manifest=manifest,
                    all_move_traces=mts,
                    all_game_traces=[gt],
                    traces_by_game={gt.game_id: mts},
                )
                assert Path(paths["move_log"]).exists()
                assert Path(paths["game_summary"]).exists()
                assert Path(paths["run_report"]).exists()
                assert Path(paths["primitive_attribution"]).exists()
                assert Path(paths["markdown_summary"]).exists()
            finally:
                os.chdir(old_cwd)

    def test_move_log_has_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "data" / "runs" / "r1"
            (run_dir / "games").mkdir(parents=True)
            (run_dir / "traces").mkdir(parents=True)
            (Path(tmpdir) / "reports" / "latest").mkdir(parents=True)
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                gt = _game_trace()
                mts = [_move_trace(centipawn_loss=50.0)]
                paths = build_reports("r1", run_dir, _manifest("r1"), mts, [gt], {gt.game_id: mts})
                df = pd.read_csv(paths["move_log"])
                assert len(df) == 1
            finally:
                os.chdir(old_cwd)

    def test_primitive_attribution_csv_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "data" / "runs" / "r2"
            (run_dir / "games").mkdir(parents=True)
            (run_dir / "traces").mkdir(parents=True)
            (Path(tmpdir) / "reports" / "latest").mkdir(parents=True)
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                gt = _game_trace()
                mts = [
                    _move_trace(centipawn_loss=float(i * 15),
                                weighted_scores={"self_material_difference": i * 0.1})
                    for i in range(1, 12)
                ]
                paths = build_reports("r2", run_dir, _manifest("r2"), mts, [gt], {gt.game_id: mts})
                df = pd.read_csv(paths["primitive_attribution"])
                assert len(df) > 0
                assert "primitive_id" in df.columns
            finally:
                os.chdir(old_cwd)


# ── ExperimentManifest is valid JSON ──────────────────────────────────────────

class TestExperimentManifest:
    def test_serializes_to_valid_json(self):
        m = _manifest()
        data = json.loads(m.to_json())
        assert data["run_id"] == "test_run"
        assert isinstance(data["game_ids"], list)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            m = _manifest("save_test")
            m.save(str(path))
            loaded = ExperimentManifest.load(str(path))
            assert loaded.run_id == "save_test"

    def test_required_fields_present(self):
        m = _manifest()
        d = json.loads(m.to_json())
        for field in ("run_id", "total_games", "player_type", "narrative_mode",
                      "completed_games", "failed_games"):
            assert field in d

    def test_defaults(self):
        m = ExperimentManifest(run_id="minimal")
        assert m.completed_games == 0
        assert m.failed_games == 0
        assert m.notes == ""

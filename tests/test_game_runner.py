"""Tests for game runner (Phase 1)."""
import pytest
import chess

from src.environment.board_manager import BoardManager
from src.environment.opponents import RandomOpponent, HeuristicOpponent
from src.gameplay.game_runner import GameRunner
from src.storage.models import GameTrace


BASE_CONFIG = {
    "game": {"max_moves": 50, "save_traces": False},
    "default_opponent": "random",
}


def test_board_manager_snapshot():
    bm = BoardManager(game_id="test-001")
    snap = bm.snapshot()
    assert snap.game_id == "test-001"
    assert snap.ply_index == 0
    assert snap.turn == "white"
    assert not snap.is_game_over
    assert len(snap.legal_moves) == 20


def test_board_manager_push_uci():
    bm = BoardManager()
    bm.push_uci("e2e4")
    assert bm.ply_index == 1
    assert bm.turn == "black"


def test_board_manager_illegal_move():
    bm = BoardManager()
    with pytest.raises(ValueError):
        bm.push_uci("e2e5")  # illegal


def test_board_manager_copy():
    bm = BoardManager()
    bm.push_uci("e2e4")
    copy = bm.copy()
    copy.push_uci("e7e5")
    assert bm.ply_index == 1  # original unchanged
    assert copy.ply_index == 2


def test_random_opponent():
    opp = RandomOpponent(seed=42)
    board = chess.Board()
    move = opp.choose_move(board)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_heuristic_opponent_prefers_captures():
    opp = HeuristicOpponent(seed=0)
    # Position where white can capture black pawn
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    # White can capture d5 with exd5
    move = opp.choose_move(board)
    assert move == "e4d5"


def test_game_runner_random_vs_random():
    config = dict(BASE_CONFIG)
    opponent = RandomOpponent(seed=7)
    runner = GameRunner(config=config, opponent=opponent, lab_color=chess.WHITE)
    trace = runner.run()
    assert isinstance(trace, GameTrace)
    assert trace.result in ("1-0", "0-1", "1/2-1/2", "*", None)  # None = max_moves hit
    assert len(trace.moves) > 0


def test_game_runner_produces_move_records():
    config = dict(BASE_CONFIG)
    opponent = RandomOpponent(seed=13)
    runner = GameRunner(config=config, opponent=opponent, lab_color=chess.WHITE)
    trace = runner.run()
    for m in trace.moves:
        assert m.uci
        assert m.san
        assert m.fen_before
        assert m.fen_after
        assert m.player in ("self", "opponent")


def test_game_trace_serialization():
    config = dict(BASE_CONFIG)
    opponent = RandomOpponent(seed=99)
    runner = GameRunner(config=config, opponent=opponent, lab_color=chess.BLACK)
    trace = runner.run()
    json_str = trace.to_json()
    restored = GameTrace.from_json(json_str)
    assert restored.game_id == trace.game_id
    assert len(restored.moves) == len(trace.moves)


def test_game_runner_heuristic_opponent():
    config = dict(BASE_CONFIG)
    opponent = HeuristicOpponent(seed=5)
    runner = GameRunner(config=config, opponent=opponent, lab_color=chess.WHITE)
    trace = runner.run()
    assert len(trace.moves) > 0


def test_game_runner_total_plies():
    config = dict(BASE_CONFIG)
    opponent = RandomOpponent(seed=1)
    runner = GameRunner(config=config, opponent=opponent)
    trace = runner.run()
    assert trace.total_plies == len(trace.moves)

"""Tests for primitive extraction (Phase 2)."""
import pytest
import chess

from src.primitives.extractor import (
    PrimitiveExtractor,
    build_default_registry,
    _extract_material_difference,
    _extract_bishop_pair,
    _extract_passed_pawns,
    _extract_isolated_pawns,
    _extract_doubled_pawns,
    _extract_hanging_own_pieces,
    _extract_castled_status,
    _extract_minor_pieces_developed,
    _extract_center_occupancy,
)
from src.primitives.registry import PrimitiveRegistry
from src.primitives.renderers import render_full_extraction, render_primitives_by_category


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_registry_builds():
    registry = build_default_registry()
    assert len(registry) >= 20


def test_registry_enable_disable():
    registry = build_default_registry()
    pid = "self_material_difference"
    registry.disable(pid)
    assert pid not in {d.primitive_id for d in registry.all_enabled()}
    registry.enable(pid)
    assert pid in {d.primitive_id for d in registry.all_enabled()}


def test_registry_by_category():
    registry = build_default_registry()
    material = registry.by_category("material")
    assert len(material) >= 3
    assert all(d.category == "material" for d in material)


# ---------------------------------------------------------------------------
# Individual extraction function tests
# ---------------------------------------------------------------------------

def test_material_difference_equal():
    board = chess.Board()
    val, norm = _extract_material_difference(board, chess.WHITE)
    assert val == 0.0
    assert abs(norm - 0.5) < 0.01


def test_material_difference_advantage():
    # Remove a black rook
    board = chess.Board()
    board.remove_piece_at(chess.A8)
    val, norm = _extract_material_difference(board, chess.WHITE)
    assert val == 5.0
    assert norm > 0.5


def test_bishop_pair_starting():
    board = chess.Board()
    has_pair, norm = _extract_bishop_pair(board, chess.WHITE)
    assert has_pair is True
    assert norm == 1.0


def test_bishop_pair_missing():
    board = chess.Board()
    board.remove_piece_at(chess.C1)
    has_pair, norm = _extract_bishop_pair(board, chess.WHITE)
    assert has_pair is False
    assert norm == 0.0


def test_passed_pawn():
    # White pawn on e5, no black pawns nearby
    board = chess.Board("8/8/8/4P3/8/8/8/8 w - - 0 1")
    count, norm = _extract_passed_pawns(board, chess.WHITE)
    assert count == 1
    assert norm > 0


def test_isolated_pawn():
    # White has only a pawn on a2 (isolated)
    board = chess.Board("8/8/8/8/8/8/P7/8 w - - 0 1")
    count, norm = _extract_isolated_pawns(board, chess.WHITE)
    assert count == 1


def test_doubled_pawn():
    # White has two pawns on e-file
    board = chess.Board("8/8/8/4P3/4P3/8/8/8 w - - 0 1")
    count, norm = _extract_doubled_pawns(board, chess.WHITE)
    assert count == 1


def test_castled_status_not_castled():
    board = chess.Board()
    castled, norm = _extract_castled_status(board, chess.WHITE)
    assert castled is False
    assert norm == 0.0


def test_castled_status_castled():
    # King on g1 = short castled
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    # Manually put king on g1 to simulate castled
    board2 = chess.Board()
    board2.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board2.remove_piece_at(chess.E1)
    castled, norm = _extract_castled_status(board2, chess.WHITE)
    assert castled is True
    assert norm == 1.0


def test_minor_pieces_developed_starting():
    board = chess.Board()
    count, norm = _extract_minor_pieces_developed(board, chess.WHITE)
    assert count == 0
    assert norm == 0.0


def test_minor_pieces_developed_after_moves():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    count, norm = _extract_minor_pieces_developed(board, chess.WHITE)
    assert count == 1


def test_center_occupancy_starting():
    board = chess.Board()
    count, norm = _extract_center_occupancy(board, chess.WHITE)
    assert count == 0


def test_hanging_own_pieces_starting():
    board = chess.Board()
    count, norm = _extract_hanging_own_pieces(board, chess.WHITE)
    assert count == 0  # Starting position has no hanging pieces


# ---------------------------------------------------------------------------
# Full extractor tests
# ---------------------------------------------------------------------------

def test_full_extraction_starting_position():
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)
    board = chess.Board()
    primitives = extractor.extract(board, chess.WHITE)
    assert len(primitives) >= 20
    for p in primitives:
        assert 0.0 <= p.normalized_value <= 1.0
        assert 0.0 <= p.confidence <= 1.0
        assert p.text_render
        assert p.primitive_id


def test_full_extraction_all_serializable():
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)
    board = chess.Board()
    primitives = extractor.extract(board, chess.WHITE)
    for p in primitives:
        data = p.model_dump()
        assert "primitive_id" in data
        assert "normalized_value" in data
        assert "text_render" in data


def test_queen_overextension_confidence():
    registry = build_default_registry()
    defn = registry.get("self_queen_overextension_risk")
    assert defn.default_confidence == 0.6


def test_render_full_extraction():
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)
    board = chess.Board()
    primitives = extractor.extract(board, chess.WHITE)
    output = render_full_extraction(primitives)
    assert "self_material_difference" in output
    assert "conf=" in output


def test_render_by_category():
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)
    board = chess.Board()
    primitives = extractor.extract(board, chess.WHITE)
    output = render_primitives_by_category(primitives)
    assert "MATERIAL" in output
    assert "KING_SAFETY" in output


def test_extraction_midgame_fen():
    registry = build_default_registry()
    extractor = PrimitiveExtractor(registry)
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(fen)
    primitives = extractor.extract(board, chess.WHITE)
    assert len(primitives) >= 20
    ids = {p.primitive_id for p in primitives}
    assert "self_minor_pieces_developed" in ids
    assert "self_castled_status" in ids

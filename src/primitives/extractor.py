from __future__ import annotations

import math
from typing import Any

import chess

from src.primitives.definitions import PrimitiveDefinition, SideScope
from src.primitives.registry import PrimitiveRegistry
from src.storage.models import PrimitiveValue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}
EXTENDED_CENTER = {chess.C3, chess.D3, chess.E3, chess.F3,
                   chess.C4, chess.D4, chess.E4, chess.F4,
                   chess.C5, chess.D5, chess.E5, chess.F5,
                   chess.C6, chess.D6, chess.E6, chess.F6}

KING_SHIELD_SQUARES_WHITE = {
    chess.G1: {chess.F2, chess.G2, chess.H2},
    chess.H1: {chess.G2, chess.H2},
    chess.C1: {chess.B2, chess.C2, chess.D2},
    chess.B1: {chess.A2, chess.B2, chess.C2},
}
KING_SHIELD_SQUARES_BLACK = {
    chess.G8: {chess.F7, chess.G7, chess.H7},
    chess.H8: {chess.G7, chess.H7},
    chess.C8: {chess.B7, chess.C7, chess.D7},
    chess.B8: {chess.A7, chess.B7, chess.C7},
}


def _material_score(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for pt, val in PIECE_VALUES.items():
        total += len(board.pieces(pt, color)) * val
    return total


def _normalize_clamp(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _is_hanging(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    """Returns True if piece on sq (of given color) is attacked and not defended."""
    piece = board.piece_at(sq)
    if piece is None or piece.color != color:
        return False
    attackers = board.attackers(not color, sq)
    if not attackers:
        return False
    defenders = board.attackers(color, sq)
    if not defenders:
        return True
    # Simple SEE-lite: if attacker value < piece value, it's effectively hanging
    min_attacker_val = min(PIECE_VALUES.get(board.piece_at(a).piece_type, 0)
                           for a in attackers if board.piece_at(a))
    piece_val = PIECE_VALUES.get(piece.piece_type, 0)
    return min_attacker_val < piece_val


# ---------------------------------------------------------------------------
# Extraction functions — each takes (board, color) where color is "self" side
# ---------------------------------------------------------------------------

def _extract_material_difference(board: chess.Board, color: chess.Color) -> tuple[float, float]:
    diff = _material_score(board, color) - _material_score(board, not color)
    normalized = _normalize_clamp(diff, -15, 15)
    return float(diff), normalized


def _extract_bishop_pair(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    has_pair = len(board.pieces(chess.BISHOP, color)) >= 2
    return has_pair, 1.0 if has_pair else 0.0


def _extract_rook_count_difference(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    diff = len(board.pieces(chess.ROOK, color)) - len(board.pieces(chess.ROOK, not color))
    return diff, _normalize_clamp(diff, -2, 2)


def _extract_queen_presence(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    present = len(board.pieces(chess.QUEEN, color)) > 0
    return present, 1.0 if present else 0.0


def _extract_minor_pieces_developed(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    developed = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if not (chess.BB_SQUARES[sq] & back_rank):
                developed += 1
    return developed, _normalize_clamp(developed, 0, 4)


def _extract_castled_status(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    # We check if king has moved from its starting square
    king_sq = board.king(color)
    if king_sq is None:
        return False, 0.0
    start_sq = chess.E1 if color == chess.WHITE else chess.E8
    # Castled = king is on g/c file on back rank
    back_rank_sqs = (chess.G1, chess.C1) if color == chess.WHITE else (chess.G8, chess.C8)
    castled = king_sq in back_rank_sqs
    return castled, 1.0 if castled else 0.0


def _extract_undeveloped_back_rank_minors(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    undeveloped = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.BB_SQUARES[sq] & back_rank:
                undeveloped += 1
    # Normalize: 0 undeveloped = 1.0 (good), 4 undeveloped = 0.0 (bad)
    return undeveloped, _normalize_clamp(4 - undeveloped, 0, 4)


def _extract_pawn_shield_quality(board: chess.Board, color: chess.Color) -> tuple[float, float]:
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0, 0.0
    shield_map = KING_SHIELD_SQUARES_WHITE if color == chess.WHITE else KING_SHIELD_SQUARES_BLACK
    shield_squares = shield_map.get(king_sq, set())
    if not shield_squares:
        return 0.5, 0.5
    present = sum(1 for sq in shield_squares
                  if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN
                  and board.piece_at(sq).color == color)
    quality = present / len(shield_squares)
    return quality, quality


def _extract_enemy_attackers_near_king(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    king_sq = board.king(color)
    if king_sq is None:
        return 0, 0.0
    # Count opponent pieces attacking squares adjacent to king
    adj = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    attacker_count = 0
    for sq in adj:
        attacker_count += len(board.attackers(not color, sq))
    # Normalize: 0 = safe (1.0), 8+ = very unsafe (0.0)
    return attacker_count, _normalize_clamp(8 - attacker_count, 0, 8)


def _extract_open_lines_toward_king(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    king_sq = board.king(color)
    if king_sq is None:
        return 0, 0.0
    king_file = chess.square_file(king_sq)
    open_count = 0
    for file_offset in (-1, 0, 1):
        f = king_file + file_offset
        if 0 <= f <= 7:
            # Check if file is open or semi-open toward king
            has_own_pawn = any(
                chess.square_file(sq) == f and board.piece_at(sq) and
                board.piece_at(sq).piece_type == chess.PAWN and
                board.piece_at(sq).color == color
                for sq in chess.SQUARES
            )
            if not has_own_pawn:
                open_count += 1
    return open_count, _normalize_clamp(3 - open_count, 0, 3)


def _extract_center_occupancy(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    count = sum(1 for sq in CENTER_SQUARES
                if board.piece_at(sq) and board.piece_at(sq).color == color)
    return count, count / 4.0


def _extract_center_attacks(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    attacks = sum(len(board.attackers(color, sq)) for sq in CENTER_SQUARES)
    return attacks, _normalize_clamp(attacks, 0, 8)


def _extract_center_pawn_presence(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    count = sum(1 for sq in CENTER_SQUARES
                if board.piece_at(sq) and
                board.piece_at(sq).piece_type == chess.PAWN and
                board.piece_at(sq).color == color)
    return count, count / 4.0


def _extract_hanging_own_pieces(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    count = sum(1 for sq in board.pieces(chess.PAWN, color) |
                board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) |
                board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color)
                if _is_hanging(board, sq, color))
    return count, _normalize_clamp(4 - count, 0, 4)


def _extract_hanging_opponent_pieces(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    opp = not color
    count = sum(1 for sq in board.pieces(chess.PAWN, opp) |
                board.pieces(chess.KNIGHT, opp) | board.pieces(chess.BISHOP, opp) |
                board.pieces(chess.ROOK, opp) | board.pieces(chess.QUEEN, opp)
                if _is_hanging(board, sq, opp))
    return count, _normalize_clamp(count, 0, 4)


def _extract_checks_available(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    if board.turn != color:
        # Simulate: generate moves that would give check
        count = 0
        for move in board.legal_moves if board.turn == color else []:
            board.push(move)
            if board.is_check():
                count += 1
            board.pop()
        return count, _normalize_clamp(count, 0, 5)
    count = 0
    for move in board.legal_moves:
        board.push(move)
        if board.is_check():
            count += 1
        board.pop()
    return count, _normalize_clamp(count, 0, 5)


def _extract_captures_available(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    if board.turn != color:
        return 0, 0.0
    count = sum(1 for m in board.legal_moves if board.is_capture(m))
    return count, _normalize_clamp(count, 0, 8)


def _extract_threatened_major_pieces(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    threatened = 0
    for pt in (chess.QUEEN, chess.ROOK):
        for sq in board.pieces(pt, color):
            if board.attackers(not color, sq):
                threatened += 1
    return threatened, _normalize_clamp(2 - threatened, 0, 2)


def _extract_knight_outpost(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    opp_pawns = board.pieces(chess.PAWN, not color)
    for sq in board.pieces(chess.KNIGHT, color):
        rank = chess.square_rank(sq)
        # Outpost: past 4th rank for color, not attackable by opp pawn
        if color == chess.WHITE and rank < 4:
            continue
        if color == chess.BLACK and rank > 3:
            continue
        file = chess.square_file(sq)
        attacked_by_pawn = False
        for pawn_sq in opp_pawns:
            pawn_file = chess.square_file(pawn_sq)
            pawn_rank = chess.square_rank(pawn_sq)
            if abs(pawn_file - file) == 1:
                if color == chess.WHITE and pawn_rank > rank:
                    attacked_by_pawn = True
                    break
                if color == chess.BLACK and pawn_rank < rank:
                    attacked_by_pawn = True
                    break
        if not attacked_by_pawn:
            return True, 1.0
    return False, 0.0


def _extract_bishop_activity(board: chess.Board, color: chess.Color) -> tuple[float, float]:
    bishops = board.pieces(chess.BISHOP, color)
    if not bishops:
        return 0.0, 0.0
    total_mobility = 0
    for sq in bishops:
        total_mobility += len(list(board.attacks(sq)))
    avg = total_mobility / len(bishops)
    return avg, _normalize_clamp(avg, 0, 13)


def _extract_rook_on_open_file(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    for sq in board.pieces(chess.ROOK, color):
        file = chess.square_file(sq)
        own_pawns_on_file = any(
            chess.square_file(psq) == file
            for psq in board.pieces(chess.PAWN, color)
        )
        if not own_pawns_on_file:
            return True, 1.0
    return False, 0.0


def _extract_queen_overextension_risk(board: chess.Board, color: chess.Color) -> tuple[float, float]:
    queens = board.pieces(chess.QUEEN, color)
    if not queens:
        return 0.0, 0.0
    queen_sq = next(iter(queens))
    queen_rank = chess.square_rank(queen_sq)
    # Overextension: queen past rank 4 (for white) / rank 3 (for black) in early game
    ply = board.ply()
    if ply > 20:  # Beyond opening, less relevant
        return 0.0, 0.0
    if color == chess.WHITE:
        risk = _normalize_clamp(queen_rank, 4, 7) if queen_rank > 3 else 0.0
    else:
        risk = _normalize_clamp(7 - queen_rank, 4, 7) if queen_rank < 4 else 0.0
    return risk, risk


def _extract_passed_pawns(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    opp_pawns = board.pieces(chess.PAWN, not color)
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked = False
        for opp_sq in opp_pawns:
            opp_file = chess.square_file(opp_sq)
            opp_rank = chess.square_rank(opp_sq)
            if abs(opp_file - file) <= 1:
                if color == chess.WHITE and opp_rank > rank:
                    blocked = True
                    break
                if color == chess.BLACK and opp_rank < rank:
                    blocked = True
                    break
        if not blocked:
            count += 1
    return count, _normalize_clamp(count, 0, 4)


def _extract_isolated_pawns(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    pawns = board.pieces(chess.PAWN, color)
    pawn_files = {chess.square_file(sq) for sq in pawns}
    isolated = sum(1 for f in pawn_files
                   if (f - 1) not in pawn_files and (f + 1) not in pawn_files)
    return isolated, _normalize_clamp(4 - isolated, 0, 4)


def _extract_doubled_pawns(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    from collections import Counter
    files = Counter(chess.square_file(sq) for sq in board.pieces(chess.PAWN, color))
    doubled = sum(v - 1 for v in files.values() if v > 1)
    return doubled, _normalize_clamp(4 - doubled, 0, 4)


def _extract_backward_pawns(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    pawns = board.pieces(chess.PAWN, color)
    pawn_files = {chess.square_file(sq): chess.square_rank(sq) for sq in pawns}
    backward = 0
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        left_support = pawn_files.get(f - 1, -1 if color == chess.WHITE else 8)
        right_support = pawn_files.get(f + 1, -1 if color == chess.WHITE else 8)
        if color == chess.WHITE:
            if r < left_support and r < right_support:
                backward += 1
        else:
            if r > left_support and r > right_support:
                backward += 1
    return backward, _normalize_clamp(4 - backward, 0, 4)


def _extract_pawn_islands(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    files = sorted({chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)})
    if not files:
        return 0, 1.0
    islands = 1
    for i in range(1, len(files)):
        if files[i] > files[i - 1] + 1:
            islands += 1
    return islands, _normalize_clamp(4 - islands, 0, 4)


def _extract_immediate_threat(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    if board.turn != color:
        return False, 0.0
    # Side to move has an immediate capture of a non-pawn piece
    for move in board.legal_moves:
        captured = board.piece_at(move.to_square)
        if captured and captured.color != color and captured.piece_type != chess.PAWN:
            return True, 1.0
    return False, 0.0


def _extract_forcing_moves_count(board: chess.Board, color: chess.Color) -> tuple[int, float]:
    if board.turn != color:
        return 0, 0.0
    count = sum(
        1 for move in board.legal_moves
        if board.is_capture(move) or board.gives_check(move)
    )
    return count, _normalize_clamp(count, 0, 8)


def _extract_tempo_gaining_candidate(board: chess.Board, color: chess.Color) -> tuple[bool, float]:
    if board.turn != color:
        return False, 0.0
    # A tempo-gaining move: attacks a higher-value piece
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        mover_val = PIECE_VALUES.get(piece.piece_type, 0)
        board.push(move)
        for opp_sq in chess.SquareSet(board.attacks(move.to_square)):
            opp_piece = board.piece_at(opp_sq)
            if opp_piece and opp_piece.color != color:
                opp_val = PIECE_VALUES.get(opp_piece.piece_type, 0)
                if opp_val > mover_val:
                    board.pop()
                    return True, 1.0
        board.pop()
    return False, 0.0


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_default_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()

    defs_and_fns: list[tuple[PrimitiveDefinition, Any]] = [
        # Material
        (PrimitiveDefinition(
            primitive_id="self_material_difference",
            name="Material Difference",
            category="material",
            description="Material advantage in pawn units (self minus opponent).",
            value_type="float",
            side_scope="self",
            render_template="Material balance: {value:+.1f} pawns.",
        ), _extract_material_difference),

        (PrimitiveDefinition(
            primitive_id="self_bishop_pair",
            name="Bishop Pair",
            category="material",
            description="Possesses both bishops.",
            value_type="bool",
            side_scope="self",
            render_template="Bishop pair: {value}.",
        ), _extract_bishop_pair),

        (PrimitiveDefinition(
            primitive_id="self_rook_count_difference",
            name="Rook Count Difference",
            category="material",
            description="Difference in rook count (self minus opponent).",
            value_type="int",
            side_scope="self",
            render_template="Rook count difference: {value:+d}.",
        ), _extract_rook_count_difference),

        (PrimitiveDefinition(
            primitive_id="self_queen_presence",
            name="Queen Presence",
            category="material",
            description="We still have our queen.",
            value_type="bool",
            side_scope="self",
            render_template="Our queen is on the board.",
        ), _extract_queen_presence),

        # Development
        (PrimitiveDefinition(
            primitive_id="self_minor_pieces_developed",
            name="Minor Pieces Developed",
            category="development",
            description="Number of knights and bishops off the back rank.",
            value_type="int",
            side_scope="self",
            render_template="{value} of our minor pieces are developed.",
        ), _extract_minor_pieces_developed),

        (PrimitiveDefinition(
            primitive_id="self_castled_status",
            name="Castled Status",
            category="development",
            description="King has castled.",
            value_type="bool",
            side_scope="self",
            render_template="King castled: {value}.",
        ), _extract_castled_status),

        (PrimitiveDefinition(
            primitive_id="self_undeveloped_back_rank_minors",
            name="Undeveloped Back-Rank Minors",
            category="development",
            description="Count of knights/bishops still on their starting squares.",
            value_type="int",
            side_scope="self",
            render_template="{value} minor piece(s) remain undeveloped on back rank.",
        ), _extract_undeveloped_back_rank_minors),

        # King safety
        (PrimitiveDefinition(
            primitive_id="self_pawn_shield_quality",
            name="Pawn Shield Quality",
            category="king_safety",
            description="Fraction of king's pawn shield intact (0-1).",
            value_type="float",
            side_scope="self",
            render_template="King pawn shield quality: {value:.0%}.",
        ), _extract_pawn_shield_quality),

        (PrimitiveDefinition(
            primitive_id="self_enemy_attackers_near_king",
            name="Enemy Attackers Near King",
            category="king_safety",
            description="Count of opponent attacks on squares adjacent to our king.",
            value_type="int",
            side_scope="self",
            render_template="{value} opponent attack(s) near our king.",
        ), _extract_enemy_attackers_near_king),

        (PrimitiveDefinition(
            primitive_id="self_open_lines_toward_king",
            name="Open Lines Toward King",
            category="king_safety",
            description="Number of open or semi-open files near our king.",
            value_type="int",
            side_scope="self",
            render_template="{value} open/semi-open file(s) near our king.",
        ), _extract_open_lines_toward_king),

        # Center
        (PrimitiveDefinition(
            primitive_id="self_center_occupancy",
            name="Center Occupancy",
            category="center_control",
            description="Number of central squares (d4/e4/d5/e5) we occupy.",
            value_type="int",
            side_scope="self",
            render_template="We occupy {value} of 4 central squares.",
        ), _extract_center_occupancy),

        (PrimitiveDefinition(
            primitive_id="self_center_attacks",
            name="Center Attacks",
            category="center_control",
            description="Number of our attacks on central squares.",
            value_type="int",
            side_scope="self",
            render_template="We attack central squares {value} times.",
        ), _extract_center_attacks),

        (PrimitiveDefinition(
            primitive_id="self_center_pawn_presence",
            name="Center Pawn Presence",
            category="center_control",
            description="Number of our pawns on central squares.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} pawn(s) in the center.",
        ), _extract_center_pawn_presence),

        # Tactical
        (PrimitiveDefinition(
            primitive_id="self_hanging_own_pieces",
            name="Hanging Own Pieces",
            category="tactical",
            description="Number of our pieces that are hanging (undefended and attacked).",
            value_type="int",
            side_scope="self",
            render_template="{value} of our piece(s) are hanging.",
        ), _extract_hanging_own_pieces),

        (PrimitiveDefinition(
            primitive_id="self_hanging_opponent_pieces",
            name="Hanging Opponent Pieces",
            category="tactical",
            description="Number of opponent pieces that are hanging.",
            value_type="int",
            side_scope="self",
            render_template="{value} opponent piece(s) are hanging.",
        ), _extract_hanging_opponent_pieces),

        (PrimitiveDefinition(
            primitive_id="self_checks_available",
            name="Checks Available",
            category="tactical",
            description="Number of legal checking moves available.",
            value_type="int",
            side_scope="self",
            render_template="{value} checking move(s) available.",
        ), _extract_checks_available),

        (PrimitiveDefinition(
            primitive_id="self_captures_available",
            name="Captures Available",
            category="tactical",
            description="Number of legal captures available.",
            value_type="int",
            side_scope="self",
            render_template="{value} capture(s) available.",
        ), _extract_captures_available),

        (PrimitiveDefinition(
            primitive_id="self_threatened_major_pieces",
            name="Threatened Major Pieces",
            category="tactical",
            description="Our queen or rooks under attack.",
            value_type="int",
            side_scope="self",
            render_template="{value} of our major piece(s) are under attack.",
        ), _extract_threatened_major_pieces),

        # Piece activity
        (PrimitiveDefinition(
            primitive_id="self_knight_outpost",
            name="Knight Outpost",
            category="piece_activity",
            description="We have a knight on an outpost square.",
            value_type="bool",
            side_scope="self",
            render_template="We have a knight on an outpost.",
        ), _extract_knight_outpost),

        (PrimitiveDefinition(
            primitive_id="self_bishop_activity",
            name="Bishop Activity",
            category="piece_activity",
            description="Average mobility (squares attacked) of our bishops.",
            value_type="float",
            side_scope="self",
            render_template="Bishop activity score: {value:.1f} squares.",
        ), _extract_bishop_activity),

        (PrimitiveDefinition(
            primitive_id="self_rook_on_open_file",
            name="Rook on Open File",
            category="piece_activity",
            description="At least one of our rooks is on an open or semi-open file.",
            value_type="bool",
            side_scope="self",
            render_template="Our rook is on an open file.",
        ), _extract_rook_on_open_file),

        (PrimitiveDefinition(
            primitive_id="self_queen_overextension_risk",
            name="Queen Overextension Risk",
            category="piece_activity",
            description="Risk that our queen is overextended in the opening.",
            value_type="float",
            side_scope="self",
            render_template="Queen overextension risk: {value:.0%}.",
            default_confidence=0.6,  # Heuristic — lower confidence per spec
        ), _extract_queen_overextension_risk),

        # Pawn structure
        (PrimitiveDefinition(
            primitive_id="self_passed_pawns",
            name="Passed Pawns",
            category="pawn_structure",
            description="Number of our passed pawns.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} passed pawn(s).",
        ), _extract_passed_pawns),

        (PrimitiveDefinition(
            primitive_id="self_isolated_pawns",
            name="Isolated Pawns",
            category="pawn_structure",
            description="Number of our isolated pawns.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} isolated pawn(s).",
        ), _extract_isolated_pawns),

        (PrimitiveDefinition(
            primitive_id="self_doubled_pawns",
            name="Doubled Pawns",
            category="pawn_structure",
            description="Number of our doubled pawns.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} doubled pawn(s).",
        ), _extract_doubled_pawns),

        (PrimitiveDefinition(
            primitive_id="self_backward_pawns",
            name="Backward Pawns",
            category="pawn_structure",
            description="Number of our backward pawns.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} backward pawn(s).",
        ), _extract_backward_pawns),

        (PrimitiveDefinition(
            primitive_id="self_pawn_islands",
            name="Pawn Islands",
            category="pawn_structure",
            description="Number of separate pawn islands.",
            value_type="int",
            side_scope="self",
            render_template="We have {value} pawn island(s).",
        ), _extract_pawn_islands),

        # Initiative
        (PrimitiveDefinition(
            primitive_id="self_immediate_threat",
            name="Immediate Threat",
            category="initiative",
            description="We can capture a non-pawn piece this move.",
            value_type="bool",
            side_scope="self",
            render_template="We have an immediate material threat.",
        ), _extract_immediate_threat),

        (PrimitiveDefinition(
            primitive_id="self_forcing_moves_count",
            name="Forcing Moves Count",
            category="initiative",
            description="Number of checks and captures available.",
            value_type="int",
            side_scope="self",
            render_template="{value} forcing move(s) (checks/captures) available.",
        ), _extract_forcing_moves_count),

        (PrimitiveDefinition(
            primitive_id="self_tempo_gaining_candidate",
            name="Tempo-Gaining Candidate",
            category="initiative",
            description="A move exists that attacks a higher-value opponent piece.",
            value_type="bool",
            side_scope="self",
            render_template="A tempo-gaining move candidate exists.",
        ), _extract_tempo_gaining_candidate),
    ]

    for defn, fn in defs_and_fns:
        registry.add(defn, fn)

    return registry


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class PrimitiveExtractor:
    def __init__(self, registry: PrimitiveRegistry):
        self.registry = registry

    def extract(self, board: chess.Board, color: chess.Color) -> list[PrimitiveValue]:
        results = []
        for defn in self.registry.all_enabled():
            if defn.extraction_fn is None:
                continue
            try:
                raw, normalized = defn.extraction_fn(board, color)
                text = defn.render(raw)
                results.append(PrimitiveValue(
                    primitive_id=defn.primitive_id,
                    name=defn.name,
                    side=defn.side_scope,
                    value=raw,
                    normalized_value=float(normalized),
                    text_render=text,
                    confidence=defn.default_confidence,
                    category=defn.category,
                ))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Extraction failed for %s: %s", defn.primitive_id, exc
                )
        return results

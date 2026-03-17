"""Test 4: One-Move Consequence.

Can the model track what happens after a single move is played?
15 samples: captures, non-captures, and special moves.
"""

import random

import chess

from .position_generator import get_simple_positions, piece_name


def _find_castling_position() -> tuple[chess.Board, chess.Move, int, str] | None:
    """Find or create a position where castling is possible."""
    # Simple kingside castling setup for white
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_castling_fen("K")
    board.turn = chess.WHITE
    if board.is_valid():
        move = chess.Move(chess.E1, chess.G1)
        return board, move, chess.G1, "white king"
    return None


def _find_en_passant_position() -> tuple[chess.Board, chess.Move, int, str] | None:
    """Create a position with en passant available."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D5, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.ep_square = chess.D6
    board.turn = chess.WHITE
    if board.is_valid():
        move = chess.Move(chess.E5, chess.D6)
        # After en passant, the white pawn is on d6
        return board, move, chess.D6, "white pawn"
    return None


def _find_promotion_position() -> tuple[chess.Board, chess.Move, int, str] | None:
    """Create a position with pawn promotion available."""
    board = chess.Board.empty()
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    if board.is_valid():
        move = chess.Move(chess.E7, chess.E8, promotion=chess.QUEEN)
        return board, move, chess.E8, "white queen"
    return None


def generate_samples(n: int = 15, seed: int = 42) -> list[dict]:
    """Generate test samples for one-move consequence test."""
    random.seed(seed)

    samples = []

    # Special moves (up to 3)
    specials = [
        ("castling", _find_castling_position),
        ("en_passant", _find_en_passant_position),
        ("promotion", _find_promotion_position),
    ]
    for move_type, gen_fn in specials:
        result = gen_fn()
        if result is None:
            continue
        board, move, target_sq, correct = result
        move_san = board.san(move)
        square_name = chess.square_name(target_sq)
        turn_color = "white" if board.turn == chess.WHITE else "black"
        samples.append({
            "fen": board.fen(),
            "move": move_san,
            "target_square": square_name,
            "correct_answer": correct,
            "move_type": move_type,
            "question": (
                f"In the position with FEN `{board.fen()}`, it is {turn_color}'s turn. "
                f"After {turn_color} plays the move {move_san}, "
                f"what piece is on square {square_name}? Answer with exactly one of: white king, "
                f"white queen, white rook, white bishop, white knight, white pawn, black king, "
                f"black queen, black rook, black bishop, black knight, black pawn, empty. "
                f"Answer only with the piece name, nothing else."
            ),
            "test": "consequences",
        })

    # Regular moves: captures and non-captures
    n_remaining = n - len(samples)
    n_captures = n_remaining // 2
    n_non_captures = n_remaining - n_captures

    boards = get_simple_positions(n * 3, seed=seed + 100)
    captures_found = 0
    non_captures_found = 0

    for board in boards:
        if captures_found >= n_captures and non_captures_found >= n_non_captures:
            break

        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)

        for move in legal_moves:
            is_capture = board.is_capture(move)

            if is_capture and captures_found >= n_captures:
                continue
            if not is_capture and non_captures_found >= n_non_captures:
                continue

            move_san = board.san(move)
            target_sq = move.to_square
            square_name = chess.square_name(target_sq)

            # Apply move to get the result
            board.push(move)
            correct = piece_name(board, target_sq)
            board.pop()

            move_type = "capture" if is_capture else "non_capture"
            turn_color = "white" if board.turn == chess.WHITE else "black"

            samples.append({
                "fen": board.fen(),
                "move": move_san,
                "target_square": square_name,
                "correct_answer": correct,
                "move_type": move_type,
                "question": (
                    f"In the position with FEN `{board.fen()}`, it is {turn_color}'s turn. "
                    f"After {turn_color} plays the move {move_san}, "
                    f"what piece is on square {square_name}? Answer with exactly one of: white king, "
                    f"white queen, white rook, white bishop, white knight, white pawn, black king, "
                    f"black queen, black rook, black bishop, black knight, black pawn, empty. "
                    f"Answer only with the piece name, nothing else."
                ),
                "test": "consequences",
            })

            if is_capture:
                captures_found += 1
            else:
                non_captures_found += 1
            break  # one sample per board

    return samples[:n]


def score_answer(raw_answer: str, correct: str) -> bool:
    """Score the model's answer about what piece is on a square after a move.

    Only counts as correct if the model produces a clear final answer.
    """
    # Reuse the strict extraction from test_fen_parsing
    from .test_fen_parsing import extract_final_answer
    final = extract_final_answer(raw_answer).lower().strip().rstrip(".")

    if final == correct:
        return True
    if correct == "empty" and final in ("empty", "no piece", "none", "empty square"):
        return True
    return False


def compute_metrics(samples: list[dict]) -> dict:
    """Compute aggregate metrics for consequences test."""
    n = len(samples)
    correct = sum(1 for s in samples if s.get("is_correct", False))

    # Breakdown by move type
    type_acc = {}
    for s in samples:
        mt = s.get("move_type", "unknown")
        type_acc.setdefault(mt, {"total": 0, "correct": 0})
        type_acc[mt]["total"] += 1
        if s.get("is_correct", False):
            type_acc[mt]["correct"] += 1

    return {
        "test_name": "One-Move Consequence",
        "num_samples": n,
        "accuracy": correct / n if n else 0,
        "baseline": 0.077,  # random from 13 options
        "type_breakdown": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in type_acc.items()
        },
    }

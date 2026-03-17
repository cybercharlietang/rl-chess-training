"""Test 2: Legal Move Generation for a specific piece.

Can the model determine the legal moves for a specific piece from a FEN position?
15 samples, simple positions.
"""

import random

import chess

from .position_generator import get_simple_positions, get_piece_with_moves, PIECE_NAMES


def generate_samples(n: int = 15, seed: int = 42) -> list[dict]:
    """Generate n test samples for legal move generation."""
    random.seed(seed)
    boards = get_simple_positions(n * 2, seed=seed)  # generate extra in case some have no good pieces
    samples = []

    for board in boards:
        if len(samples) >= n:
            break

        result = get_piece_with_moves(board)
        if result is None:
            continue

        sq, piece_type, moves = result
        square_name = chess.square_name(sq)
        piece_type_name = PIECE_NAMES[piece_type]
        piece_color = "white" if board.turn == chess.WHITE else "black"

        # Get legal moves in SAN for this specific piece
        legal_sans = []
        for m in moves:
            legal_sans.append(board.san(m))

        question = (
            f"In the position with FEN `{board.fen()}`, it is {piece_color}'s turn. "
            f"List ALL legal moves for the {piece_color} {piece_type_name} on {square_name}. "
            f"Use SAN notation. List only the moves, separated by commas. "
            f"Do not include moves for any other piece."
        )

        samples.append({
            "fen": board.fen(),
            "square": square_name,
            "piece_type": piece_type_name,
            "correct_moves": legal_sans,
            "question": question,
            "test": "legal_moves",
        })

    return samples


def parse_move_list(raw_answer: str, board: chess.Board = None, piece_square: int = None) -> list[str]:
    """Parse a comma-separated list of moves from model output.

    Looks for the final answer after </think> tags first, then falls back
    to the last line. Also normalizes square-only notation (e.g. 'a6') to
    proper SAN by checking against legal moves.
    """
    import re

    # 1. Try to get answer after </think>
    answer = ""
    if "</think>" in raw_answer:
        after = raw_answer.split("</think>")[-1].strip()
        answer = after

    # 2. Try <answer> tags
    if not answer:
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_answer, re.DOTALL)
        if m:
            answer = m.group(1).strip()

    # 3. Fall back to last few lines
    if not answer:
        lines = [l.strip() for l in raw_answer.strip().split("\n") if l.strip()]
        answer = "\n".join(lines[-5:]) if lines else ""

    # Split by comma, semicolon, newline, "and", or spaces between move-like tokens
    parts = re.split(r"[,;\n]|\band\b", answer)
    raw_moves = []
    for p in parts:
        p = p.strip().rstrip(".")
        if not p:
            continue

        # Short enough to be a move directly
        if len(p) <= 10:
            p = re.sub(r"^\d+[\.\)]\s*", "", p).strip()
            p = re.sub(r"^[-•]\s*", "", p).strip()
            if p:
                raw_moves.append(p)
            continue

        # Longer text — extract move-like tokens from it
        # Match SAN patterns (e.g., Nf3, Bxe5+, O-O, e4, exd5) or square names
        move_tokens = re.findall(
            r"\b([KQRBN]?[a-h]?x?[a-h][1-8][+#]?|O-O(?:-O)?)\b", p
        )
        raw_moves.extend(move_tokens)

    # Normalize: try to match raw squares to proper SAN
    normalized = []
    for m in raw_moves:
        # Already valid SAN?
        try:
            board.parse_san(m)
            normalized.append(m)
            continue
        except (chess.InvalidMoveError, chess.IllegalMoveError,
                chess.AmbiguousMoveError, ValueError):
            pass

        # Try as coordinate notation (e.g., "h7h6" or "a7a5")
        if len(m) == 4 and m[:2].isalpha() is False:
            try:
                move = chess.Move.from_uci(m)
                if move in board.legal_moves:
                    normalized.append(board.san(move))
                    continue
            except (ValueError, chess.InvalidMoveError):
                pass

        # Try as destination square (e.g., "a6" for a pawn move)
        if len(m) == 2 and m[0] in "abcdefgh" and m[1] in "12345678":
            target_sq = chess.parse_square(m)
            # Find legal moves to this square from our piece
            if piece_square is not None:
                for legal in board.legal_moves:
                    if legal.from_square == piece_square and legal.to_square == target_sq:
                        normalized.append(board.san(legal))
                        break
                else:
                    # Maybe it's a valid move from any piece
                    for legal in board.legal_moves:
                        if legal.to_square == target_sq:
                            normalized.append(board.san(legal))
                            break

    return normalized if normalized else raw_moves


def score_answer(raw_answer: str, correct_moves: list[str], board: chess.Board,
                 piece_square: int = None) -> dict:
    """Score model's move list against correct legal moves."""
    predicted = parse_move_list(raw_answer, board, piece_square)

    correct_set = set(correct_moves)
    predicted_set = set(predicted)

    # Check which predicted moves are actually legal for any piece
    valid_sans = set()
    for m in predicted:
        try:
            board.parse_san(m)
            valid_sans.add(m)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass

    true_positives = correct_set & predicted_set
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    recall = len(true_positives) / len(correct_set) if correct_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "predicted_moves": predicted,
        "true_positives": list(true_positives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_invalid_san": len(predicted_set - valid_sans),
        "num_wrong_piece_moves": len(valid_sans - correct_set),
    }


def compute_metrics(samples: list[dict]) -> dict:
    """Compute aggregate metrics for legal move generation test."""
    n = len(samples)
    precisions = [s["score"]["precision"] for s in samples if "score" in s]
    recalls = [s["score"]["recall"] for s in samples if "score" in s]
    f1s = [s["score"]["f1"] for s in samples if "score" in s]

    return {
        "test_name": "Legal Move Generation",
        "num_samples": n,
        "mean_precision": sum(precisions) / len(precisions) if precisions else 0,
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0,
        "mean_f1": sum(f1s) / len(f1s) if f1s else 0,
        "baseline": 0.0,  # random would produce garbage
    }

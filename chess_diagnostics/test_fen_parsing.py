"""Test 1: FEN → Piece Identification.

Can the model decode FEN notation to identify what piece is on a given square?
25 samples, simple positions with few pieces.
"""

import random

import chess

from .position_generator import get_simple_positions, piece_name


def generate_samples(n: int = 25, seed: int = 42) -> list[dict]:
    """Generate n test samples for FEN piece identification."""
    random.seed(seed)
    boards = get_simple_positions(n, seed=seed)
    samples = []

    for board in boards:
        # Pick a random square — bias toward occupied squares (70/30)
        occupied = list(board.piece_map().keys())
        empty = [s for s in range(64) if s not in occupied]

        if random.random() < 0.7 and occupied:
            square = random.choice(occupied)
        elif empty:
            square = random.choice(empty)
        else:
            square = random.choice(occupied)

        square_name = chess.square_name(square)
        correct = piece_name(board, square)

        question = (
            f"In the position with FEN `{board.fen()}`, what piece is on square {square_name}? "
            f"Answer with exactly one of: white king, white queen, white rook, white bishop, "
            f"white knight, white pawn, black king, black queen, black rook, black bishop, "
            f"black knight, black pawn, empty. Answer only with the piece name, nothing else."
        )

        samples.append({
            "fen": board.fen(),
            "square": square_name,
            "correct_answer": correct,
            "question": question,
            "test": "fen_parsing",
        })

    return samples


def score_answer(raw_answer: str, correct: str) -> bool:
    """Score a model's answer against the correct piece name.

    Only counts as correct if the model produces a clear final answer,
    not just mentions the correct piece during reasoning.
    """
    final = extract_final_answer(raw_answer).lower().strip().rstrip(".")

    if final == correct:
        return True

    # Handle common variations of the correct answer
    if correct == "empty" and final in ("empty", "no piece", "none", "empty square", "nothing"):
        return True

    return False


def extract_final_answer(raw: str) -> str:
    """Extract only the final answer, not mid-reasoning text.

    Looks for explicit answer markers. If the model never concludes,
    returns empty string (no credit for partial reasoning).
    """
    import re

    # 1. Check for <answer> tags
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Check for text after </think>
    if "</think>" in raw:
        after = raw.split("</think>")[-1].strip()
        # Take first non-empty line after </think>
        for line in after.split("\n"):
            line = line.strip()
            if line:
                return line

    # 3. Check for a concluding statement pattern
    patterns = [
        r"(?:the answer is|answer:|the piece (?:on \w+ )?is|so,? (?:the )?(?:piece )?(?:on \w+ )?is|therefore,? (?:the )?(?:piece )?is)\s*[:\-]?\s*(.+?)\.?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE | re.MULTILINE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            # Only accept if it looks like a piece name, not mid-reasoning
            valid_answers = ["white king", "white queen", "white rook", "white bishop",
                           "white knight", "white pawn", "black king", "black queen",
                           "black rook", "black bishop", "black knight", "black pawn", "empty"]
            if candidate.lower() in valid_answers:
                return candidate

    # No clear final answer found — model didn't finish reasoning
    return ""


def compute_metrics(samples: list[dict]) -> dict:
    """Compute aggregate metrics for FEN parsing test."""
    n = len(samples)
    correct = sum(1 for s in samples if s.get("is_correct", False))

    # Breakdown by piece type
    piece_types = {}
    for s in samples:
        pt = s["correct_answer"]
        piece_types.setdefault(pt, {"total": 0, "correct": 0})
        piece_types[pt]["total"] += 1
        if s.get("is_correct", False):
            piece_types[pt]["correct"] += 1

    piece_breakdown = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in piece_types.items()
    }

    return {
        "test_name": "FEN → Piece Identification",
        "num_samples": n,
        "accuracy": correct / n if n else 0,
        "baseline": 0.077,  # random from 13 options
        "piece_breakdown": piece_breakdown,
    }

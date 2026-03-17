"""Test 3: Move Legality Judgment.

Can the model judge whether a specific move is legal in a given position?
25 samples (13 legal, 12 illegal with plausible-looking moves).
"""

import random

import chess

from .position_generator import get_simple_positions, generate_plausible_illegal_move


def generate_samples(n: int = 25, seed: int = 42) -> list[dict]:
    """Generate n test samples for move legality judgment."""
    random.seed(seed)
    n_legal = (n + 1) // 2  # 13
    n_illegal = n - n_legal  # 12

    boards = get_simple_positions(n * 3, seed=seed)  # extra for failures
    samples = []
    legal_count = 0
    illegal_count = 0

    random.shuffle(boards)

    for board in boards:
        if len(samples) >= n:
            break

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            continue

        turn_color = "white" if board.turn == chess.WHITE else "black"

        if legal_count < n_legal:
            # Generate a legal move sample
            move = random.choice(legal_moves)
            move_san = board.san(move)
            samples.append({
                "fen": board.fen(),
                "move": move_san,
                "is_legal": True,
                "correct_answer": "yes",
                "category": "legal",
                "question": (
                    f"In the position with FEN `{board.fen()}`, it is {turn_color}'s turn. "
                    f"Is the move {move_san} a legal move for {turn_color}? "
                    f"Answer only 'yes' or 'no'."
                ),
                "test": "legality",
            })
            legal_count += 1

        elif illegal_count < n_illegal:
            # Generate a plausible illegal move
            result = generate_plausible_illegal_move(board)
            if result is None:
                continue
            illegal_san, category = result
            samples.append({
                "fen": board.fen(),
                "move": illegal_san,
                "is_legal": False,
                "correct_answer": "no",
                "category": category,
                "question": (
                    f"In the position with FEN `{board.fen()}`, it is {turn_color}'s turn. "
                    f"Is the move {illegal_san} a legal move for {turn_color}? "
                    f"Answer only 'yes' or 'no'."
                ),
                "test": "legality",
            })
            illegal_count += 1

    # Shuffle so legal/illegal are interleaved
    random.shuffle(samples)
    return samples[:n]


def score_answer(raw_answer: str, correct: str) -> bool:
    """Score yes/no answer."""
    from .model_utils import extract_short_answer
    answer = extract_short_answer(raw_answer).lower().strip().rstrip(".")

    if correct == "yes":
        return answer in ("yes", "yes, it is legal", "yes, the move is legal", "legal")
    else:
        return answer in ("no", "no, it is not legal", "no, the move is not legal",
                          "illegal", "not legal", "no, it is illegal")


def compute_metrics(samples: list[dict]) -> dict:
    """Compute metrics for legality judgment test."""
    n = len(samples)
    correct = sum(1 for s in samples if s.get("is_correct", False))

    # Precision/recall for detecting illegal moves
    true_pos = sum(1 for s in samples if not s["is_legal"] and s.get("is_correct", False))
    false_neg = sum(1 for s in samples if not s["is_legal"] and not s.get("is_correct", False))
    false_pos = sum(1 for s in samples if s["is_legal"] and not s.get("is_correct", False))
    true_neg = sum(1 for s in samples if s["is_legal"] and s.get("is_correct", False))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Breakdown by category
    category_acc = {}
    for s in samples:
        cat = s.get("category", "unknown")
        category_acc.setdefault(cat, {"total": 0, "correct": 0})
        category_acc[cat]["total"] += 1
        if s.get("is_correct", False):
            category_acc[cat]["correct"] += 1

    return {
        "test_name": "Move Legality Judgment",
        "num_samples": n,
        "accuracy": correct / n if n else 0,
        "baseline": 0.50,
        "precision_illegal": precision,
        "recall_illegal": recall,
        "f1_illegal": f1,
        "category_breakdown": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in category_acc.items()
        },
    }

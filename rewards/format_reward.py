"""Move extraction and legality reward.

DeepSeek-R1-Distill uses native <think>...</think> reasoning. The move
is whatever comes after </think> — no <answer> tags needed.
"""

import re

import chess


def extract_move(text: str) -> str | None:
    """Extract the predicted move from model output.

    Looks for text after </think>. Falls back to last token if no
    </think> tag is found (e.g. if model skips reasoning).
    """
    # Primary: extract text after </think>
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        raw = lines[-1] if lines else ""

    if not raw:
        return None

    # Take first whitespace-delimited token, strip punctuation
    tokens = raw.split()
    if not tokens:
        return None
    move = tokens[0].rstrip(".,;!?:")
    return move if move else None


# Keep old name as alias for backward compat with evaluate.py
extract_answer_move = extract_move


def is_legal_move(text: str, fen: str) -> float:
    """Check that the extracted move is legal. Returns 1.0 or 0.0."""
    move_san = extract_move(text)
    if move_san is None:
        return 0.0
    board = chess.Board(fen)
    try:
        move = board.parse_san(move_san)
        return 1.0 if move in board.legal_moves else 0.0
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return 0.0


def has_valid_tags(text: str) -> float:
    """Check output has reasoning (</think> present)."""
    return 1.0 if "</think>" in text else 0.0


def format_reward(text: str, fen: str) -> float:
    """Combined: has reasoning + legal move. No language check needed."""
    tag_score = has_valid_tags(text)
    legal_score = is_legal_move(text, fen)
    return (tag_score + legal_score) / 2.0

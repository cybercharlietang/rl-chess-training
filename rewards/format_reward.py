"""Format compliance rewards: tags, language, legality."""

import re

import chess
from langdetect import detect, LangDetectException


def has_valid_tags(text: str) -> float:
    """Check that output contains reasoning and answer tags.

    Accepts two patterns:
      1. <think>...</think> ... <answer>...</answer>  (full tags in output)
      2. ...</think> ... <answer>...</answer>  (opening <think> was in the
         generation prompt and stripped during decoding)
    """
    has_think_block = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    has_think_close = bool(re.search(r"</think>", text))
    has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
    return 1.0 if ((has_think_block or has_think_close) and has_answer) else 0.0


def is_english(text: str) -> float:
    """Check that the text is in English."""
    # Extract text outside tags for language detection
    clean = re.sub(r"<[^>]+>", " ", text).strip()
    if len(clean) < 10:
        # Too short to detect reliably — give benefit of the doubt
        return 1.0
    try:
        return 1.0 if detect(clean) == "en" else 0.0
    except LangDetectException:
        return 0.0


def is_legal_move(text: str, fen: str) -> float:
    """Check that the move inside <answer> tags is a legal SAN move."""
    move_san = extract_answer_move(text)
    if move_san is None:
        return 0.0
    board = chess.Board(fen)
    try:
        move = board.parse_san(move_san)
        return 1.0 if move in board.legal_moves else 0.0
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return 0.0


def format_reward(text: str, fen: str) -> float:
    """Combined format reward (average of three checks)."""
    tag_score = has_valid_tags(text)
    lang_score = is_english(text)
    legal_score = is_legal_move(text, fen)
    return (tag_score + lang_score + legal_score) / 3.0


def extract_answer_move(text: str) -> str | None:
    """Extract the move from <answer> tags in model output.

    Handles various edge cases:
      - Multiple <answer> tags (takes the last one)
      - Extra whitespace
      - Move with trailing punctuation
    """
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not matches:
        return None

    # Take the last match (model might self-correct)
    raw = matches[-1].strip()

    # Clean: take first whitespace-delimited token, strip punctuation
    tokens = raw.split()
    if not tokens:
        return None
    move = tokens[0].rstrip(".,;!?")
    return move if move else None

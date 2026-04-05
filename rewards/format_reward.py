"""Move extraction and legality reward.

DeepSeek-R1-Distill uses native <think>...</think> reasoning. The move
is whatever comes after </think> — no <answer> tags needed.
"""

import re

import chess


def _looks_like_san(token: str) -> bool:
    """Quick check if a token could be a SAN chess move."""
    # SAN moves: e4, Nf3, Bxb5, O-O, Qxf4+, Rxf7#, etc.
    clean = token.strip("*.,;!?:#+ ")
    if not clean:
        return False
    if clean in ("O-O", "O-O-O"):
        return True
    # Must start with piece letter or pawn file
    if clean[0] in "KQRBNabcdefgh" and len(clean) >= 2:
        return True
    return False


def extract_move(text: str) -> str | None:
    """Extract the predicted move from model output.

    The model outputs <think>reasoning</think> then an answer which may be:
      - Just the move: "Rxf7"
      - Prose: "The best move is Rxf7"
      - Markdown bold: "**Rxf7**"
      - Multi-line with explanation

    Strategy:
      1. Get text after </think> (or full text if no tag)
      2. Try to find a SAN-looking token via multiple heuristics
    """
    # Get text after </think>
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        raw = lines[-1] if lines else ""

    if not raw:
        return None

    # Strategy 1: bold markdown move like **Rxf7** or **Bxb5**
    bold_matches = re.findall(r'\*\*([^*]+)\*\*', raw)
    for bm in bold_matches:
        clean = bm.strip().rstrip(".,;!?:")
        if _looks_like_san(clean):
            return clean

    # Strategy 2: "is <SAN move>" pattern (e.g. "The best move is Bc6.")
    m = re.search(r'\bis\s+([KQRBN]?[a-h]?[x]?[a-h][1-8][+#=]?(?:[QRBN])?)', raw)
    if m:
        return m.group(1).rstrip(".,;!?:")

    # Strategy 3: castling
    if "O-O-O" in raw:
        return "O-O-O"
    if "O-O" in raw:
        return "O-O"

    # Strategy 4: first SAN-looking token on its own line (standalone move)
    for line in raw.splitlines():
        line = line.strip().replace("**", "")
        if line and _looks_like_san(line.rstrip(".,;!?:")):
            return line.rstrip(".,;!?:")

    # Strategy 5: scan all tokens for first SAN match
    # Use stricter SAN regex: piece + optional file + optional x + square + optional promo/check
    san_re = re.compile(r'^[KQRBN]?[a-h]?x?[a-h][1-8](?:=[QRBN])?[+#]?$')
    for token in raw.split():
        clean = token.strip("*.,;!?:()\"'")
        if san_re.match(clean):
            return clean

    return None


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

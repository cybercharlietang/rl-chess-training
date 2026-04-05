"""Prompt templates for DeepSeek-R1-Distill-Qwen-14B chess puzzles.

Uses native <think> reasoning. Move is extracted from text after </think>.
"""

SYSTEM_PROMPT = (
    "You are a professional chess player. Analyze the position, then output "
    "only the best move in SAN notation.\n\n"
    "Reminder of chess rules:\n"
    "- Bishops move diagonally.\n"
    "- Rooks move horizontally or vertically.\n"
    "- Knights jump in an L-shape.\n"
    "- Queens combine rook and bishop movement.\n"
    "- Kings move one square in any direction.\n"
    "- Pawns move forward, capture diagonally, and can promote."
)


def build_user_message(fen: str, legal_moves: list[str]) -> str:
    """Build the user turn content for a chess puzzle."""
    moves_str = ", ".join(legal_moves)
    return (
        f"The current FEN is {fen} and legal moves are {moves_str}. "
        f"What is the best move?"
    )


def build_chat_messages(fen: str, legal_moves: list[str]) -> list[dict]:
    """Build the full chat message list for tokenizer.apply_chat_template().

    DeepSeek-R1-Distill uses native <think> reasoning. We let it think
    naturally, then extract the move from text after </think>.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(fen, legal_moves)},
    ]

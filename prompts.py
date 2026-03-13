"""Prompt templates for Qwen3-8B-Instruct chess puzzles."""

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant is a professional "
    "chess player who first thinks about the reasoning process and then provides "
    "the answer.\n\n"
    "The reasoning must be in <think> </think> tags. The answer must be in "
    "<answer> </answer> tags.\n\n"
    "The answer must be a single move in SAN notation from the legal moves list.\n\n"
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

    Returns a list of {role, content} dicts ready for the Qwen3 chat template.
    We use the system prompt as a 'system' role message, then a 'user' message.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(fen, legal_moves)},
    ]

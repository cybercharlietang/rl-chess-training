"""Test 5: Declarative Chess Rules Knowledge.

Can the model answer factual questions about chess rules?
No FEN parsing required — pure knowledge test.
50 questions with unambiguous correct answers.
"""


def generate_samples(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate chess rules knowledge questions."""
    # All questions have short, unambiguous answers
    all_questions = [
        # Piece movement (10)
        {"question": "In chess, how does a knight move?",
         "correct_answer": "l-shape",
         "accept": ["l-shape", "l shape", "l shaped", "two squares in one direction and one square perpendicular",
                     "2 squares and 1 square", "two and one", "l-shaped pattern"]},
        {"question": "In chess, can a bishop move horizontally?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can a rook move diagonally?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can a queen move diagonally?",
         "correct_answer": "yes",
         "accept": ["yes"]},
        {"question": "In chess, how many squares can a king move in one turn?",
         "correct_answer": "one",
         "accept": ["one", "1", "one square", "1 square", "one (except castling)"]},
        {"question": "In chess, can a pawn move backwards?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can a knight jump over other pieces?",
         "correct_answer": "yes",
         "accept": ["yes"]},
        {"question": "In chess, can a rook jump over other pieces?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, how does a pawn capture?",
         "correct_answer": "diagonally",
         "accept": ["diagonally", "diagonal", "diagonally forward", "one square diagonally forward",
                     "one square diagonally"]},
        {"question": "In chess, can a bishop change the color of squares it operates on during a game?",
         "correct_answer": "no",
         "accept": ["no"]},

        # Special moves (10)
        {"question": "In chess, what is castling?",
         "correct_answer": "king moves two squares toward a rook",
         "accept": ["king moves two squares", "king and rook swap", "king moves two squares toward a rook",
                     "king moves two squares towards the rook"]},
        {"question": "In chess, can you castle if your king has already moved?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can you castle if the rook has already moved?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can you castle while in check?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can you castle if the king would pass through a square that is attacked?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, what is en passant?",
         "correct_answer": "a pawn captures another pawn that just moved two squares",
         "accept": ["pawn capture", "pawn captures a pawn that moved two squares", "special pawn capture",
                     "a pawn captures another pawn that just advanced two squares"]},
        {"question": "In chess, what happens when a pawn reaches the opposite end of the board?",
         "correct_answer": "promotion",
         "accept": ["promotion", "it promotes", "it is promoted", "it gets promoted",
                     "it can be promoted to another piece", "pawn promotion"]},
        {"question": "In chess, can a pawn promote to a king?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, can a pawn promote to a knight?",
         "correct_answer": "yes",
         "accept": ["yes"]},
        {"question": "In chess, on its first move, how many squares can a pawn advance?",
         "correct_answer": "one or two",
         "accept": ["one or two", "1 or 2", "two", "2", "up to two", "either one or two"]},

        # Board setup (8)
        {"question": "In chess, how many squares are on the board?",
         "correct_answer": "64",
         "accept": ["64", "sixty-four", "sixty four"]},
        {"question": "In chess, how many pawns does each player start with?",
         "correct_answer": "8",
         "accept": ["8", "eight"]},
        {"question": "In chess, how many knights does each player start with?",
         "correct_answer": "2",
         "accept": ["2", "two"]},
        {"question": "In chess, how many bishops does each player start with?",
         "correct_answer": "2",
         "accept": ["2", "two"]},
        {"question": "In chess, how many queens does each player start with?",
         "correct_answer": "1",
         "accept": ["1", "one"]},
        {"question": "In chess, which piece starts on d1 in the initial position?",
         "correct_answer": "white queen",
         "accept": ["white queen", "queen", "the white queen"]},
        {"question": "In chess, which piece starts on e1 in the initial position?",
         "correct_answer": "white king",
         "accept": ["white king", "king", "the white king"]},
        {"question": "In chess, what color is the square a1?",
         "correct_answer": "dark",
         "accept": ["dark", "black", "dark square"]},

        # Check and checkmate (8)
        {"question": "In chess, what is check?",
         "correct_answer": "the king is under attack",
         "accept": ["king is under attack", "king is attacked", "king is threatened",
                     "the king is under attack", "when the king is attacked"]},
        {"question": "In chess, what is checkmate?",
         "correct_answer": "the king is in check and cannot escape",
         "accept": ["king cannot escape check", "king is in check and cannot escape",
                     "the king is in check and has no legal moves", "no legal moves to escape check"]},
        {"question": "In chess, can you move into check?",
         "correct_answer": "no",
         "accept": ["no"]},
        {"question": "In chess, what is stalemate?",
         "correct_answer": "no legal moves but not in check",
         "accept": ["no legal moves but not in check", "player has no legal moves and is not in check",
                     "draw", "a draw", "the player to move has no legal moves but is not in check"]},
        {"question": "In chess, is stalemate a win or a draw?",
         "correct_answer": "draw",
         "accept": ["draw", "a draw", "it is a draw"]},
        {"question": "In chess, how many ways can you respond to a check?",
         "correct_answer": "three",
         "accept": ["three", "3", "block, capture, or move the king",
                     "move the king, block, or capture the attacking piece"]},
        {"question": "In chess, can a king capture a piece that is giving check?",
         "correct_answer": "yes",
         "accept": ["yes", "yes, if it is safe"]},
        {"question": "In chess, can two kings be adjacent to each other?",
         "correct_answer": "no",
         "accept": ["no"]},

        # FEN notation (6)
        {"question": "In FEN notation, what does an uppercase letter represent?",
         "correct_answer": "a white piece",
         "accept": ["white piece", "a white piece", "white pieces", "a piece belonging to white"]},
        {"question": "In FEN notation, what does a lowercase letter represent?",
         "correct_answer": "a black piece",
         "accept": ["black piece", "a black piece", "black pieces", "a piece belonging to black"]},
        {"question": "In FEN notation, what does the number 8 in the piece placement mean?",
         "correct_answer": "8 empty squares",
         "accept": ["8 empty squares", "eight empty squares", "an entire rank of empty squares",
                     "all squares are empty", "the entire rank is empty"]},
        {"question": "In FEN notation, what does the letter 'N' represent?",
         "correct_answer": "white knight",
         "accept": ["white knight", "a white knight", "knight"]},
        {"question": "In FEN notation, what does the letter 'k' represent?",
         "correct_answer": "black king",
         "accept": ["black king", "a black king"]},
        {"question": "In FEN notation, does the piece placement start from rank 8 or rank 1?",
         "correct_answer": "rank 8",
         "accept": ["rank 8", "8", "from rank 8", "the 8th rank", "eighth rank"]},

        # Game rules (8)
        {"question": "In chess, which player moves first?",
         "correct_answer": "white",
         "accept": ["white"]},
        {"question": "In chess, what is the most powerful piece?",
         "correct_answer": "queen",
         "accept": ["queen", "the queen"]},
        {"question": "In chess, what is the least valuable piece (excluding the king)?",
         "correct_answer": "pawn",
         "accept": ["pawn", "the pawn"]},
        {"question": "In chess, approximately how many points is a knight worth?",
         "correct_answer": "3",
         "accept": ["3", "three", "3 points", "three points"]},
        {"question": "In chess, approximately how many points is a rook worth?",
         "correct_answer": "5",
         "accept": ["5", "five", "5 points", "five points"]},
        {"question": "In chess, approximately how many points is a queen worth?",
         "correct_answer": "9",
         "accept": ["9", "nine", "9 points", "nine points"]},
        {"question": "In chess, can a game end in a draw?",
         "correct_answer": "yes",
         "accept": ["yes"]},
        {"question": "In chess, what happens if the same position occurs three times?",
         "correct_answer": "draw by threefold repetition",
         "accept": ["draw", "threefold repetition", "draw by repetition",
                     "a draw can be claimed", "the game can be drawn"]},
    ]

    # Append instruction to each question
    for q in all_questions:
        q["question"] = q["question"] + " Answer concisely in one sentence."
        q["test"] = "rules_knowledge"

    return all_questions[:n]


def score_answer(raw_answer: str, sample: dict) -> bool:
    """Score a rules knowledge answer."""
    from .model_utils import extract_short_answer
    answer = extract_short_answer(raw_answer).lower().strip().rstrip(".")

    accept_list = [a.lower() for a in sample["accept"]]

    # Check if any accepted answer is contained in the model's response
    for accepted in accept_list:
        if accepted in answer:
            return True

    # For yes/no questions, also check the full raw output after </think>
    if sample["correct_answer"] in ("yes", "no"):
        # Look for a clear yes/no in the final part
        import re
        raw_lower = raw_answer.lower()
        # Check after </think> if present
        if "</think>" in raw_lower:
            after_think = raw_lower.split("</think>")[-1].strip()
        else:
            after_think = raw_lower

        # For yes/no, check the last line
        lines = [l.strip() for l in after_think.split("\n") if l.strip()]
        if lines:
            last = lines[-1].rstrip(".")
            if sample["correct_answer"] == "yes" and ("yes" in last and "no" not in last):
                return True
            if sample["correct_answer"] == "no" and ("no" in last and "yes" not in last):
                return True

    return False


def compute_metrics(samples: list[dict]) -> dict:
    """Compute aggregate metrics for rules knowledge test."""
    n = len(samples)
    correct = sum(1 for s in samples if s.get("is_correct", False))

    # Breakdown by category
    categories = {}
    cat_map = {
        0: "Piece Movement", 10: "Special Moves", 20: "Board Setup",
        28: "Check/Checkmate", 36: "FEN Notation", 42: "Game Rules",
    }
    for i, s in enumerate(samples):
        cat = "Other"
        for start_idx, name in sorted(cat_map.items()):
            if i >= start_idx:
                cat = name
        categories.setdefault(cat, {"total": 0, "correct": 0})
        categories[cat]["total"] += 1
        if s.get("is_correct", False):
            categories[cat]["correct"] += 1

    return {
        "test_name": "Declarative Rules Knowledge",
        "num_samples": n,
        "accuracy": correct / n if n else 0,
        "baseline": 0.0,
        "category_breakdown": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0
            for k, v in categories.items()
        },
    }

"""Generate test positions for diagnostic tests.

Focuses on simple positions with few pieces to test baseline chess understanding.
"""

import json
import os
import random

import chess


PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}

COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}


def piece_name(board: chess.Board, square: int) -> str:
    """Return human-readable piece name like 'white knight' for a square."""
    piece = board.piece_at(square)
    if piece is None:
        return "empty"
    return f"{COLOR_NAMES[piece.color]} {PIECE_NAMES[piece.piece_type]}"


def generate_simple_endgame() -> chess.Board:
    """Generate a random simple endgame position (3-8 pieces)."""
    board = chess.Board.empty()

    # Always place kings
    white_king_sq = random.choice(range(64))
    board.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))

    # Place black king at least 2 squares away
    for _ in range(100):
        black_king_sq = random.choice(range(64))
        if black_king_sq != white_king_sq and chess.square_distance(white_king_sq, black_king_sq) >= 2:
            board.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
            break

    # Add 1-6 random pieces
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    num_extra = random.randint(1, 6)
    for _ in range(num_extra):
        sq = random.choice([s for s in range(64) if board.piece_at(s) is None])
        pt = random.choice(piece_types)
        color = random.choice([chess.WHITE, chess.BLACK])
        # Don't place pawns on rank 1 or 8
        rank = chess.square_rank(sq)
        if pt == chess.PAWN and rank in (0, 7):
            pt = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
        board.set_piece_at(sq, chess.Piece(pt, color))

    # Set side to move
    board.turn = random.choice([chess.WHITE, chess.BLACK])

    # Validate — if not legal, retry
    if board.is_valid() and list(board.legal_moves):
        return board
    return generate_simple_endgame()


def load_puzzle_positions(data_dir: str = "data", max_positions: int = 100) -> list[str]:
    """Load FEN positions from Lichess puzzle data if available."""
    path = os.path.join(data_dir, "eval.jsonl")
    if not os.path.exists(path):
        return []
    fens = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            fens.append(sample["fen"])
            if len(fens) >= max_positions:
                break
    return fens


def get_simple_positions(n: int, seed: int = 42) -> list[chess.Board]:
    """Get n simple positions, preferring generated endgames for simplicity."""
    random.seed(seed)
    boards = []

    # Try puzzle positions first, filter for simple ones (< 12 pieces)
    puzzle_fens = load_puzzle_positions()
    simple_puzzles = []
    for fen in puzzle_fens:
        board = chess.Board(fen)
        piece_count = len(board.piece_map())
        if piece_count <= 12:
            simple_puzzles.append(board)

    # Use simple puzzles if we have enough, otherwise generate endgames
    if len(simple_puzzles) >= n:
        random.shuffle(simple_puzzles)
        boards = simple_puzzles[:n]
    else:
        boards = list(simple_puzzles)
        while len(boards) < n:
            boards.append(generate_simple_endgame())

    return boards


def generate_plausible_illegal_move(board: chess.Board) -> tuple[str, str] | None:
    """Generate a plausible-looking but illegal move and categorize it.

    Returns (move_san, category) or None if we can't generate one.
    Categories: blocked_path, friendly_square, into_check, wrong_pattern, pinned
    """
    piece_map = board.piece_map()

    # Strategy 1: Move a piece to a square occupied by a friendly piece
    for sq, piece in piece_map.items():
        if piece.color != board.turn:
            continue
        # Find a friendly piece's square that this piece could "move to" geometrically
        for target_sq, target_piece in piece_map.items():
            if target_piece.color == board.turn and target_sq != sq:
                # Construct a pseudo-move
                move = chess.Move(sq, target_sq)
                if move not in board.legal_moves:
                    try:
                        board.push(move)
                        board.pop()
                        # If push didn't raise, it's actually pseudo-legal
                        san = board.san(move) if move in board.pseudo_legal_moves else None
                        if san:
                            return san, "friendly_square"
                    except (ValueError, AssertionError):
                        pass

    # Strategy 2: Move a piece in a pattern it can't do
    for sq, piece in piece_map.items():
        if piece.color != board.turn:
            continue
        if piece.piece_type == chess.KNIGHT:
            # Try a diagonal move (knights don't move diagonally)
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            for df, dr in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nf, nr = file + df, rank + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    target = chess.square(nf, nr)
                    if board.piece_at(target) is None:
                        move = chess.Move(sq, target)
                        if move not in board.legal_moves:
                            piece_name_str = chess.piece_name(piece.piece_type).capitalize()
                            sq_name = chess.square_name(sq)
                            target_name = chess.square_name(target)
                            return f"{piece_name_str[0]}{target_name}", "wrong_pattern"

        if piece.piece_type == chess.PAWN:
            # Try moving pawn backwards
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            direction = -1 if piece.color == chess.WHITE else 1
            nr = rank + direction
            if 0 <= nr < 8:
                target = chess.square(file, nr)
                if board.piece_at(target) is None:
                    target_name = chess.square_name(target)
                    return target_name, "wrong_pattern"

    # Strategy 3: Just pick a random non-legal move that looks vaguely chess-like
    for sq, piece in piece_map.items():
        if piece.color != board.turn:
            continue
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        for df in range(-2, 3):
            for dr in range(-2, 3):
                if df == 0 and dr == 0:
                    continue
                nf, nr = file + df, rank + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    target = chess.square(nf, nr)
                    move = chess.Move(sq, target)
                    if move not in board.legal_moves and board.piece_at(target) is None:
                        if piece.piece_type == chess.PAWN:
                            return chess.square_name(target), "wrong_pattern"
                        else:
                            prefix = chess.piece_symbol(piece.piece_type).upper()
                            if prefix == "P":
                                prefix = ""
                            return f"{prefix}{chess.square_name(target)}", "wrong_pattern"

    return None


def get_piece_with_moves(board: chess.Board) -> tuple[int, chess.PieceType, list[chess.Move]] | None:
    """Find a piece on the board that has legal moves, return (square, piece_type, moves)."""
    piece_map = board.piece_map()
    candidates = []
    for sq, piece in piece_map.items():
        if piece.color != board.turn:
            continue
        moves = [m for m in board.legal_moves if m.from_square == sq]
        if moves:
            candidates.append((sq, piece.piece_type, moves))
    if not candidates:
        return None
    return random.choice(candidates)

"""Dense reward using Stockfish centipawn evaluation."""

import math

import chess
import chess.engine

from config import Config


def create_engine(config: Config) -> chess.engine.SimpleEngine:
    """Create a Stockfish engine instance."""
    return chess.engine.SimpleEngine.popen_uci(config.stockfish_path)


def dense_stockfish_reward(
    fen: str,
    predicted_move_san: str,
    engine: chess.engine.SimpleEngine,
    depth: int = 15,
) -> float:
    """Score a predicted move using Stockfish evaluation.

    Process:
      1. Parse FEN, validate the move is legal
      2. Apply the move to get new position
      3. Run Stockfish eval at given depth
      4. Normalize centipawn score to [0, 1] via sigmoid(cp / 400)
      5. If Black to move in original position, negate cp before sigmoid

    Returns:
      Reward in [0, 1], or -1.0 if the move is illegal.
    """
    board = chess.Board(fen)

    # Try to parse the predicted move
    try:
        move = board.parse_san(predicted_move_san)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
        return -1.0

    if move not in board.legal_moves:
        return -1.0

    # Who is moving
    player_is_white = board.turn == chess.WHITE

    # Apply move and evaluate resulting position
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()  # always from White's perspective

    # Convert to centipawns
    if score.is_mate():
        mate_in = score.mate()
        # mate_in >= 0 means White mates in N (0 = already checkmate)
        # mate_in < 0 means White gets mated in |N| moves
        cp = (10000 - abs(mate_in)) if mate_in >= 0 else -(10000 - abs(mate_in))
    else:
        cp = score.score()

    # If player is Black, a good move for Black means a negative cp (from White's view)
    # We want the reward to be high when the player's move is good, so negate for Black
    if not player_is_white:
        cp = -cp

    # Sigmoid normalization to [0, 1]
    reward = 1.0 / (1.0 + math.exp(-cp / 400.0))
    return reward

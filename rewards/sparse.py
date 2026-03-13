"""Sparse reward: binary match against the puzzle solution."""


def sparse_reward(predicted_move: str, solution_move: str) -> float:
    """Return 1.0 if predicted move matches solution, 0.0 otherwise."""
    return 1.0 if predicted_move.strip() == solution_move.strip() else 0.0

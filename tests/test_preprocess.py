"""Tests for the data preprocessing pipeline."""

import chess
import pytest

from data.preprocess import decompose_puzzle


# Constructed puzzle: Black plays Na5 (blunder), White mates with Qxf7#
SIMPLE_PUZZLE = {
    "PuzzleId": "test001",
    "FEN": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4",
    "Moves": "c6a5 h5f7",
    "Rating": "1500",
    "RatingDeviation": "100",
    "Popularity": "90",
    "NbPlays": "100",
    "Themes": "mate mateIn1",
    "GameUrl": "https://lichess.org/abc",
    "OpeningTags": "",
}

# Multi-move puzzle: opponent setup + 3 solution moves (player, opponent, player)
MULTI_MOVE_PUZZLE = {
    "PuzzleId": "multi1",
    "FEN": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "Moves": "e7e5 d2d4 e5d4 d1d4",
    "Rating": "800",
    "RatingDeviation": "100",
    "Popularity": "80",
    "NbPlays": "50",
    "Themes": "opening",
    "GameUrl": "https://lichess.org/def",
    "OpeningTags": "",
}


def test_simple_puzzle_decomposition():
    """A 1-move puzzle should produce exactly 1 training sample."""
    samples = decompose_puzzle(SIMPLE_PUZZLE)
    assert len(samples) == 1

    s = samples[0]
    assert s["puzzle_rating"] == 1500
    assert s["puzzle_id"] == "test001"
    assert s["solution_move"] in s["legal_moves"], (
        f"Solution {s['solution_move']} not in legal moves {s['legal_moves']}"
    )

    # Verify the FEN is valid and it's the player's turn
    board = chess.Board(s["fen"])
    assert board.is_valid()


def test_multi_move_puzzle_decomposition():
    """A multi-move puzzle should produce samples only for the player's moves."""
    samples = decompose_puzzle(MULTI_MOVE_PUZZLE)

    # Moves after setup: d2d4 (player, idx1), e5d4 (opponent, idx2), d1d4 (player, idx3)
    # Player's moves are at odd indices: idx1 (d2d4) and idx3 (d1d4) → 2 samples
    assert len(samples) == 2

    for s in samples:
        assert s["solution_move"] in s["legal_moves"]
        board = chess.Board(s["fen"])
        assert board.is_valid()


def test_puzzle_legal_moves_are_complete():
    """Legal moves list should contain all legal moves in the position."""
    samples = decompose_puzzle(SIMPLE_PUZZLE)
    s = samples[0]
    board = chess.Board(s["fen"])

    expected_legal = sorted([board.san(m) for m in board.legal_moves])
    actual_legal = sorted(s["legal_moves"])
    assert actual_legal == expected_legal


def test_san_format():
    """Moves should be in SAN format, not UCI."""
    samples = decompose_puzzle(SIMPLE_PUZZLE)
    s = samples[0]
    # SAN moves don't look like "e2e4" (4 chars, all lowercase)
    for move in s["legal_moves"]:
        # UCI pattern: exactly 4-5 lowercase chars like "e2e4" or "e7e8q"
        assert not (len(move) in (4, 5) and move.isalnum() and move == move.lower()), (
            f"Move '{move}' looks like UCI, should be SAN"
        )

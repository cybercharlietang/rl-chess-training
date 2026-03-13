"""Tests for reward functions."""

import math

import chess
import chess.engine
import pytest

from rewards.sparse import sparse_reward
from rewards.format_reward import (
    extract_answer_move,
    has_valid_tags,
    is_english,
    is_legal_move,
    format_reward,
)
from rewards.dense_stockfish import create_engine, dense_stockfish_reward
from config import Config


# ── Sparse reward ────────────────────────────────────────────────────────

class TestSparseReward:
    def test_exact_match(self):
        assert sparse_reward("Nf3", "Nf3") == 1.0

    def test_mismatch(self):
        assert sparse_reward("Nf3", "e4") == 0.0

    def test_whitespace_tolerance(self):
        assert sparse_reward(" Nf3 ", "Nf3") == 1.0


# ── Answer extraction ────────────────────────────────────────────────────

class TestExtractAnswerMove:
    def test_simple(self):
        assert extract_answer_move("<think>blah</think><answer>Nf3</answer>") == "Nf3"

    def test_with_whitespace(self):
        assert extract_answer_move("<answer> Nf3 </answer>") == "Nf3"

    def test_multiple_tags_takes_last(self):
        text = "<answer>e4</answer> wait no <answer>Nf3</answer>"
        assert extract_answer_move(text) == "Nf3"

    def test_no_tags(self):
        assert extract_answer_move("I think Nf3 is best") is None

    def test_empty_answer(self):
        assert extract_answer_move("<answer>  </answer>") is None

    def test_trailing_punctuation(self):
        assert extract_answer_move("<answer>Nf3.</answer>") == "Nf3"

    def test_multiple_words_takes_first(self):
        assert extract_answer_move("<answer>Nf3 is the best move</answer>") == "Nf3"


# ── Tag validation ───────────────────────────────────────────────────────

class TestHasValidTags:
    def test_valid(self):
        assert has_valid_tags("<think>reasoning</think><answer>Nf3</answer>") == 1.0

    def test_missing_think(self):
        assert has_valid_tags("<answer>Nf3</answer>") == 0.0

    def test_missing_answer(self):
        assert has_valid_tags("<think>reasoning</think>") == 0.0

    def test_no_tags(self):
        assert has_valid_tags("just some text") == 0.0


# ── Legal move check ────────────────────────────────────────────────────

class TestIsLegalMove:
    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_legal_move(self):
        assert is_legal_move("<answer>e4</answer>", self.START_FEN) == 1.0

    def test_illegal_move(self):
        assert is_legal_move("<answer>e5</answer>", self.START_FEN) == 0.0

    def test_no_answer_tag(self):
        assert is_legal_move("e4", self.START_FEN) == 0.0

    def test_nonsense_move(self):
        assert is_legal_move("<answer>xyz</answer>", self.START_FEN) == 0.0


# ── Language detection ───────────────────────────────────────────────────

class TestIsEnglish:
    def test_english(self):
        assert is_english("I think the best move is to control the center.") == 1.0

    def test_short_text_passes(self):
        # Short text gets benefit of the doubt
        assert is_english("Nf3") == 1.0


# ── Dense Stockfish reward ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Shared Stockfish engine for all dense reward tests."""
    config = Config()
    eng = create_engine(config)
    yield eng
    eng.quit()


class TestDenseStockfishReward:
    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_illegal_move_returns_negative(self, engine):
        """Illegal move (e5 as White from start) should return -1.0."""
        assert dense_stockfish_reward(self.START_FEN, "e5", engine, depth=10) == -1.0

    def test_nonsense_move_returns_negative(self, engine):
        """Unparseable move should return -1.0."""
        assert dense_stockfish_reward(self.START_FEN, "xyz", engine, depth=10) == -1.0

    def test_legal_move_returns_in_unit_interval(self, engine):
        """Any legal move should return a reward in [0, 1]."""
        reward = dense_stockfish_reward(self.START_FEN, "e4", engine, depth=10)
        assert 0.0 <= reward <= 1.0

    def test_good_opening_move_above_half(self, engine):
        """e4 from starting position is a strong move; reward should be > 0.5.
        The sigmoid normalizes around 0 cp = 0.5, and White's best moves
        from the start give a small positive centipawn advantage."""
        reward = dense_stockfish_reward(self.START_FEN, "e4", engine, depth=10)
        assert reward > 0.5

    def test_reward_ordering(self, engine):
        """A strong opening move (e4) should score higher than a weak one (h4).
        This isn't guaranteed at very low depth, but depth 10 should be enough."""
        reward_e4 = dense_stockfish_reward(self.START_FEN, "e4", engine, depth=10)
        reward_h4 = dense_stockfish_reward(self.START_FEN, "h4", engine, depth=10)
        assert reward_e4 > reward_h4

    def test_black_perspective(self, engine):
        """When it's Black's turn, a good Black move should get a high reward.
        After 1. e4, Black plays e5 (solid) vs h5 (weak). e5 should score higher."""
        # Position after 1. e4
        black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reward_e5 = dense_stockfish_reward(black_fen, "e5", engine, depth=10)
        reward_h5 = dense_stockfish_reward(black_fen, "h5", engine, depth=10)
        assert 0.0 <= reward_e5 <= 1.0
        assert 0.0 <= reward_h5 <= 1.0
        assert reward_e5 > reward_h5

    def test_black_good_move_above_half(self, engine):
        """A solid Black reply should score around 0.5 (roughly equal position)."""
        black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        reward = dense_stockfish_reward(black_fen, "e5", engine, depth=10)
        # After 1. e4 e5 the position is roughly equal, reward ≈ 0.5
        assert reward > 0.4

    def test_mate_in_one_white(self, engine):
        """White to move with mate in 1 (Scholar's mate: Qxf7#).
        The mating move should give a reward very close to 1.0."""
        mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        reward = dense_stockfish_reward(mate_fen, "Qxf7+", engine, depth=10)
        assert reward > 0.99

    def test_winning_position_high_reward(self, engine):
        """In a position where White is up a queen, a reasonable move
        should score well. Sigmoid(cp/400) compresses large advantages,
        so we check > 0.65 rather than > 0.9."""
        winning_fen = "r5k1/8/8/8/8/8/8/3Q3K w - - 0 1"
        reward = dense_stockfish_reward(winning_fen, "Qd7", engine, depth=10)
        assert reward > 0.65

    def test_sigmoid_normalization_math(self, engine):
        """Verify the sigmoid formula: reward = 1 / (1 + exp(-cp/400)).
        At cp=0, reward should be exactly 0.5.
        At cp=400, reward should be ~0.731."""
        # cp=0 → 0.5
        assert abs(1.0 / (1.0 + math.exp(0)) - 0.5) < 1e-10
        # cp=400 → sigmoid(1) ≈ 0.731
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert abs(expected - 0.7310585786300049) < 1e-10
        # cp=-400 → sigmoid(-1) ≈ 0.269
        expected_neg = 1.0 / (1.0 + math.exp(1.0))
        assert abs(expected_neg - 0.2689414213699951) < 1e-10

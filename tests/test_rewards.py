"""Tests for reward functions."""

import chess
import pytest

from rewards.sparse import sparse_reward
from rewards.format_reward import (
    extract_answer_move,
    has_valid_tags,
    is_english,
    is_legal_move,
    format_reward,
)


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

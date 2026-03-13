"""Tests for prompt construction."""

from prompts import build_chat_messages, build_user_message, SYSTEM_PROMPT


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
LEGAL_MOVES = ["e4", "d4", "Nf3", "Nc3"]


def test_user_message_contains_fen():
    msg = build_user_message(START_FEN, LEGAL_MOVES)
    assert START_FEN in msg


def test_user_message_contains_all_legal_moves():
    msg = build_user_message(START_FEN, LEGAL_MOVES)
    for move in LEGAL_MOVES:
        assert move in msg


def test_chat_messages_structure():
    messages = build_chat_messages(START_FEN, LEGAL_MOVES)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_prompt_has_think_answer_instructions():
    assert "<think>" in SYSTEM_PROMPT
    assert "<answer>" in SYSTEM_PROMPT


def test_system_prompt_has_chess_rules():
    assert "Bishops" in SYSTEM_PROMPT
    assert "Knights" in SYSTEM_PROMPT

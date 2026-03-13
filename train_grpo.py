"""Main GRPO training script using TRL's GRPOTrainer.

Skeleton — GPU-dependent parts are stubbed out.
"""

import argparse
import json
import math
import re

import chess

from config import Config
from prompts import build_chat_messages
from rewards.format_reward import extract_answer_move, format_reward
from rewards.sparse import sparse_reward

# Dense reward import is deferred since it needs Stockfish binary


def load_training_data(path: str) -> list[dict]:
    """Load training samples from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def build_reward_function(config: Config):
    """Build the combined reward function for GRPO.

    Returns a callable that takes (completions, prompts_metadata) and
    returns a list of float rewards.

    The exact signature will depend on TRL's GRPOTrainer API.
    """
    if config.reward_mode == "dense":
        from rewards.dense_stockfish import create_engine, dense_stockfish_reward
        engine = create_engine(config)

    def reward_fn(completions: list[str], metadata: list[dict]) -> list[float]:
        rewards = []
        for completion, meta in zip(completions, metadata):
            fen = meta["fen"]
            solution = meta["solution_move"]

            # Extract predicted move
            predicted = extract_answer_move(completion)

            # Move reward
            if predicted is None:
                move_r = -1.0
            elif config.reward_mode == "sparse":
                move_r = sparse_reward(predicted, solution)
            else:
                move_r = dense_stockfish_reward(fen, predicted, engine, config.stockfish_depth)

            # Format reward
            fmt_r = format_reward(completion, fen)

            # Legal move reward
            legal_r = 0.0
            if predicted:
                board = chess.Board(fen)
                try:
                    m = board.parse_san(predicted)
                    legal_r = 1.0 if m in board.legal_moves else 0.0
                except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                    legal_r = 0.0

            total = (config.lambda_move * move_r
                     + config.lambda_fmt * fmt_r
                     + config.lambda_legal * legal_r)
            rewards.append(total)

        return rewards

    return reward_fn


# ── GPU-dependent: model loading, GRPOTrainer setup, training loop ───────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_mode", type=str, default="dense",
                        choices=["sparse", "dense"])
    parser.add_argument("--config_overrides", type=str, default=None,
                        help="JSON string of config overrides")
    args = parser.parse_args()

    config = Config(reward_mode=args.reward_mode)

    print(f"Training config: reward_mode={config.reward_mode}")
    print(f"Model: {config.model_name}")

    samples = load_training_data(config.train_data_path)
    print(f"Loaded {len(samples)} training samples.")

    # TODO (GPU phase):
    # 1. Load model + tokenizer (Qwen3-8B-Instruct)
    # 2. Apply LoRA via peft
    # 3. Build prompts using build_chat_messages + tokenizer.apply_chat_template
    # 4. Initialize GRPOTrainer with model, reward_fn, config
    # 5. trainer.train()
    # 6. Save LoRA adapter
    print("Model loading/training not yet implemented (needs GPU).")

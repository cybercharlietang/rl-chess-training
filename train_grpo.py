"""GRPO training script for chess puzzle solving using TRL's GRPOTrainer."""

import argparse
import json
import math
import os
import time

import chess
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from config import Config
from prompts import SYSTEM_PROMPT, build_user_message
from rewards.format_reward import (
    extract_answer_move,
    format_reward,
    has_valid_tags,
    is_legal_move,
)
from rewards.sparse import sparse_reward


# ── Reward functions ─────────────────────────────────────────────────────

def make_move_reward_fn(config: Config):
    """Build the move correctness reward function.

    For sparse: exact match with solution.
    For dense: Stockfish evaluation of the predicted move.
    """
    if config.reward_mode == "dense":
        from rewards.dense_stockfish import create_engine, dense_stockfish_reward
        engine = create_engine(config)

    def reward_fn(prompts, completions, fen, solution_move, **kwargs):
        rewards = []
        for completion, f, solution in zip(completions, fen, solution_move):
            # Extract text from completion (may be list of message dicts)
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            predicted = extract_answer_move(text)

            if predicted is None:
                rewards.append(-1.0)
            elif config.reward_mode == "sparse":
                rewards.append(sparse_reward(predicted, solution))
            else:
                rewards.append(dense_stockfish_reward(f, predicted, engine, config.stockfish_depth))
        return rewards

    return reward_fn


def format_reward_fn(prompts, completions, fen, **kwargs):
    """Format compliance reward: valid tags + English + legal move."""
    rewards = []
    for completion, f in zip(completions, fen):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        rewards.append(format_reward(text, f))
    return rewards


def legal_move_reward_fn(prompts, completions, fen, **kwargs):
    """Legal move reward: 1.0 if predicted move is legal, 0.0 otherwise."""
    rewards = []
    for completion, f in zip(completions, fen):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        predicted = extract_answer_move(text)
        if predicted is None:
            rewards.append(0.0)
        else:
            board = chess.Board(f)
            try:
                m = board.parse_san(predicted)
                rewards.append(1.0 if m in board.legal_moves else 0.0)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                rewards.append(0.0)
    return rewards


# ── Dataset ──────────────────────────────────────────────────────────────

def build_dataset(data_path: str, tokenizer) -> Dataset:
    """Load training data and build a HuggingFace Dataset with chat prompts."""
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    rows = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(s["fen"], s["legal_moves"])},
        ]
        rows.append({
            "prompt": messages,
            "fen": s["fen"],
            "solution_move": s["solution_move"],
            "puzzle_rating": s["puzzle_rating"],
        })

    return Dataset.from_list(rows)


# ── Logging callback ────────────────────────────────────────────────────

class CompletionLogCallback:
    """Logs sample completions periodically for the visualizer."""

    def __init__(self, log_path: str, log_every_steps: int = 50):
        self.log_path = log_path
        self.log_every = log_every_steps
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass  # Training metrics are handled by GRPOTrainer's built-in logging


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_mode", type=str, default="dense",
                        choices=["sparse", "dense"])
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps (default: train for full epochs)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Override training data path")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume training from a checkpoint directory")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Load a trained LoRA adapter before training (continues from finetuned weights)")
    args = parser.parse_args()

    config = Config(reward_mode=args.reward_mode)
    if args.train_data:
        config.train_data_path = args.train_data
    output_dir = args.output_dir or os.path.join(config.output_dir, "grpo_run")

    print(f"{'='*60}")
    print(f"GRPO Training")
    print(f"{'='*60}")
    print(f"  Model:          {config.model_name}")
    print(f"  Reward mode:    {config.reward_mode}")
    print(f"  LoRA rank:      {config.lora_rank}")
    print(f"  Batch size:     {config.batch_size}")
    print(f"  Num generations:{config.num_generations}")
    print(f"  Learning rate:  {config.learning_rate}")
    print(f"  Output dir:     {output_dir}")
    print(f"{'='*60}")

    # ── Load tokenizer ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build dataset ────────────────────────────────────────────────
    train_dataset = build_dataset(config.train_data_path, tokenizer)
    eval_dataset = build_dataset(config.eval_data_path, tokenizer)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # ── LoRA config ──────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )

    # ── Reward functions ─────────────────────────────────────────────
    move_reward = make_move_reward_fn(config)
    reward_funcs = [move_reward, format_reward_fn, legal_move_reward_fn]
    reward_weights = [config.lambda_move, config.lambda_fmt, config.lambda_legal]

    # ── GRPO config ──────────────────────────────────────────────────
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        max_steps=args.max_steps or -1,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        seed=config.seed,
        # GRPO-specific
        num_generations=config.num_generations,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        reward_weights=reward_weights,
        log_completions=True,
        num_completions_to_print=3,
        # Logging & saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        report_to="none",
        # Eval
        eval_strategy="steps",
        eval_steps=config.save_steps,
        per_device_eval_batch_size=config.batch_size,
    )

    # ── Load model ────────────────────────────────────────────────────
    if args.adapter:
        # Load base model + trained LoRA adapter, then continue training
        from peft import PeftModel
        print(f"Loading base model {config.model_name} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16,
        )
        print(f"Loading adapter from {args.adapter} ...")
        model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
        model_or_name = model
        peft_cfg = None  # LoRA already applied
    else:
        model_or_name = config.model_name
        peft_cfg = peft_config

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_or_name,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/3600:.1f} hours.")

    # ── Save ─────────────────────────────────────────────────────────
    trainer.save_model(os.path.join(output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"Saved final adapter to {output_dir}/final_adapter")


if __name__ == "__main__":
    main()

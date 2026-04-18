"""GRPO training v2 — new reward: stockfish + illegal penalty + babble penalty.

Layout (5x H200):
  GPU 0:     vLLM server (TP=1)
  GPUs 1-4:  DDP training (G=4, 16 completions/step)
"""

# Shim for peft+transformers compatibility
import transformers.integrations.tensor_parallel as _tp
if not hasattr(_tp, "EmbeddingParallel"):
    _tp.EmbeddingParallel = _tp.RowwiseParallel

import argparse, json, os, time, dataclasses
import chess, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from config import Config
from prompts import SYSTEM_PROMPT, build_user_message
from rewards.format_reward import extract_move


class MetricsLogger(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.step_start = None
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        entry = {"step": state.global_step, "time": time.time(),
                 "step_time_s": (time.time() - self.step_start) if self.step_start else None,
                 **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))}}
        if torch.cuda.is_available():
            peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
            entry["gpu_mem_peak_gb"] = peak / 1024**3
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# === REWARD FUNCTIONS v2 ===

def make_stockfish_reward_fn(config):
    """Move quality via Stockfish. Returns -1 for illegal/missing moves."""
    from rewards.dense_stockfish import create_engine, dense_stockfish_reward
    engine = create_engine(config)

    def reward_fn(prompts, completions, fen, **kwargs):
        rewards = []
        for completion, f in zip(completions, fen):
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            predicted = extract_move(text)
            if predicted is None:
                rewards.append(-1.0)
                continue
            board = chess.Board(f)
            try:
                move = board.parse_san(predicted)
                if move not in board.legal_moves:
                    rewards.append(-1.0)  # illegal — can't eval
                else:
                    rewards.append(dense_stockfish_reward(f, predicted, engine, config.stockfish_depth))
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                rewards.append(-1.0)
        return rewards
    return reward_fn


def illegal_penalty_fn(prompts, completions, fen, **kwargs):
    """-1 if move is illegal or missing, 0 if legal."""
    rewards = []
    for completion, f in zip(completions, fen):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        predicted = extract_move(text)
        if predicted is None:
            rewards.append(-1.0)
            continue
        board = chess.Board(f)
        try:
            move = board.parse_san(predicted)
            rewards.append(0.0 if move in board.legal_moves else -1.0)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            rewards.append(-1.0)
    return rewards


def babble_penalty_fn(prompts, completions, **kwargs):
    """-0.2 if model finished thinking but keeps talking. No penalty for genuine thinking."""
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        if "</think>" in text:
            post_answer = text.split("</think>")[-1]
            if len(post_answer.split()) > 50:
                rewards.append(-0.2)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)  # no penalty for genuine thinking (no </think> yet)
    return rewards


def build_dataset(data_path):
    rows = []
    with open(data_path) as f:
        for line in f:
            s = json.loads(line)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(s["fen"], s["legal_moves"])},
                ],
                "fen": s["fen"],
                "solution_move": s["solution_move"],
                "puzzle_rating": s["puzzle_rating"],
            })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_mode", choices=["sparse", "dense"], default=None)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--num_generations", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--train_data", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--run_name", type=str, default="grpo_v2")
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--adapter", type=str, default=None)
    args = ap.parse_args()

    config = Config()
    if args.reward_mode: config.reward_mode = args.reward_mode
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.num_generations is not None: config.num_generations = args.num_generations
    if args.max_new_tokens is not None: config.max_new_tokens = args.max_new_tokens
    if args.train_data: config.train_data_path = args.train_data
    if args.save_steps is not None: config.save_steps = args.save_steps

    output_dir = args.output_dir or os.path.join(config.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "step_logs.jsonl")

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    n_gpus = torch.cuda.device_count()
    print(f"\n=== GRPO v2 ({n_gpus} DDP GPUs + external vLLM) ===")
    print(f"  Model:        {config.model_name}")
    print(f"  LR:           {config.learning_rate}")
    print(f"  G:            {config.num_generations}")
    print(f"  batch/GPU:    {config.per_device_train_batch_size}")
    print(f"  max_tokens:   {config.max_new_tokens}")
    print(f"  max_steps:    {args.max_steps}")
    print(f"  save_steps:   {config.save_steps}")
    print(f"  train_data:   {config.train_data_path}")
    print(f"  Reward:       stockfish(1.0) + illegal(-1.0) + babble(-0.2)")
    print(f"  output_dir:   {output_dir}")

    print(f"\nLoading {config.model_name} in bf16 ...")
    model_or_name = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(config.train_data_path)
    print(f"  Train size:   {len(train_ds)}")

    peft_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules, task_type="CAUSAL_LM",
    )

    # v2 reward: stockfish + illegal penalty + babble penalty
    stockfish_reward = make_stockfish_reward_fn(config)
    reward_funcs = [stockfish_reward, illegal_penalty_fn, babble_penalty_fn]
    reward_weights = [1.0, 1.0, 1.0]

    peft_cfg = peft_config
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter} …")
        model_or_name = PeftModel.from_pretrained(model_or_name, args.adapter, is_trainable=True)
        peft_cfg = None

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        max_steps=args.max_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=config.bf16,
        seed=config.seed,
        ddp_find_unused_parameters=False,
        num_generations=config.num_generations,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        reward_weights=reward_weights,
        beta=config.kl_penalty_coeff,
        log_completions=config.log_completions,
        num_completions_to_print=config.num_completions_to_print,
        use_vllm=config.use_vllm,
        vllm_mode="server",
        vllm_server_host=config.vllm_server_host,
        vllm_server_port=config.vllm_server_port,
        vllm_importance_sampling_correction=False,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=True,
        report_to="none",
        save_strategy="steps",
        eval_strategy="no",
    )

    logger = MetricsLogger(log_path)

    trainer = GRPOTrainer(
        model=model_or_name,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        callbacks=[logger],
    )

    print("\nStarting training ...\n")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")

    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final adapter saved to {final_dir}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

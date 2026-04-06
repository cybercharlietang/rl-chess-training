"""All hyperparameters in one place."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO — sized for 8x H100 SXM
    # 1 prompt × 4 completions per GPU, 8 GPUs = 8 prompts, 32 completions per step
    num_generations: int = 4
    max_new_tokens: int = 4096  # 8192 OOMs with DDP, 4096 fits in 80GB H100
    temperature: float = 0.7
    clip_range: float = 0.2
    kl_penalty_coeff: float = 0.0  # no reference model — saves 28GB/GPU

    # Training
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 4  # must be divisible by num_generations (4/4=1 prompt/GPU)
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Reward — dense Stockfish + legality, no format reward
    reward_mode: Literal["sparse", "dense"] = "dense"
    stockfish_depth: int = 15
    stockfish_path: str = "/usr/games/stockfish"
    lambda_move: float = 1.0
    lambda_legal: float = 0.5

    # Data
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_train_samples: int = 7000
    num_eval_samples: int = 3000
    puzzle_rating_min: int = 200
    puzzle_rating_max: int = 2800

    # Output
    output_dir: str = "outputs"
    save_steps: int = 10
    logging_steps: int = 1

    # Reproducibility
    seed: int = 42

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

    # GRPO — sized for 5x H200 (143 GB each) with DDP training + external vLLM
    num_generations: int = 4
    max_new_tokens: int = 8192
    temperature: float = 0.7
    clip_range: float = 0.2
    kl_penalty_coeff: float = 0.0

    # Training
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True

    # vLLM server
    use_vllm: bool = True
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85

    # Reward
    reward_mode: Literal["sparse", "dense"] = "dense"
    stockfish_depth: int = 15
    stockfish_path: str = "/usr/games/stockfish"
    lambda_move: float = 1.0
    lambda_legal: float = 0.5

    # Data
    train_data_path: str = "data/train_easy.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    num_train_samples: int = 7000
    num_eval_samples: int = 3000
    puzzle_rating_min: int = 200
    puzzle_rating_max: int = 2800

    # Output & logging
    output_dir: str = "outputs"
    save_steps: int = 10
    logging_steps: int = 1
    save_total_limit: int = 3

    # Reproducibility
    seed: int = 42

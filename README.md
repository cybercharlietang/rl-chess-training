# Chess-GRPO

Training LLMs to play chess using GRPO (Group Relative Policy Optimization) on Lichess puzzles. Replicates [Chess-R1](https://arxiv.org/abs/2507.00726) using TRL's `GRPOTrainer`.

## Results

Base model: **DeepSeek-R1-Distill-Qwen-7B** with LoRA (rank 64).

| Metric | Baseline | After 16 GRPO steps |
|--------|----------|-------------------|
| Puzzle accuracy | 6% | 9% |
| Legal move rate | 66% | 53% |
| Format compliance | 69% | 61% |

Training reward improved from 0.84 → 1.20 over 16 steps with dense Stockfish rewards. The main bottleneck is ~30% of completions truncating due to FEN parsing loops before producing an answer.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
apt-get install stockfish
```

## Usage

### 1. Download and preprocess data
```bash
python -m data.download       # Downloads Lichess puzzle database (~1GB)
python -m data.preprocess     # Creates data/train.jsonl (7k) and data/eval.jsonl (3k)
```

### 2. Baseline evaluation
```bash
python evaluate.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --batch_size 16
```

### 3. GRPO training
```bash
# Dense Stockfish rewards (recommended)
python train_grpo.py --reward_mode dense

# Sparse rewards (exact match only)
python train_grpo.py --reward_mode sparse

# Short test run
python train_grpo.py --reward_mode dense --max_steps 16
```

### 4. Evaluate trained model
```bash
python evaluate.py --model outputs/grpo_run/final_adapter --batch_size 16
```

### 5. Visualize results
```bash
pip install streamlit
streamlit run visualizer.py
```

## Architecture

```
├── config.py                 # All hyperparameters
├── prompts.py                # Chat prompt template with <think>/<answer> tags
├── train_grpo.py             # GRPO training (TRL GRPOTrainer + LoRA)
├── evaluate.py               # Evaluation on held-out puzzles
├── visualizer.py             # Streamlit app for inspecting results
├── data/
│   ├── download.py           # Download Lichess puzzle CSV
│   └── preprocess.py         # Convert to (FEN, legal_moves, solution) pairs
├── rewards/
│   ├── sparse.py             # Binary reward: correct move or not
│   ├── dense_stockfish.py    # Stockfish eval → sigmoid-normalized [0,1]
│   └── format_reward.py      # Tag compliance + English + legal move check
└── tests/                    # 39 tests covering rewards, data, prompts
```

## Reward Function

Combined reward = `λ_move * move_reward + λ_fmt * format_reward + λ_legal * legal_reward`

- **Move reward (dense):** Apply predicted move, run Stockfish depth 15, normalize centipawn score via sigmoid(cp/400)
- **Format reward:** Average of tag compliance, English detection, legal move check
- **Legal reward:** 1.0 if predicted move is legal in the position, 0.0 otherwise

Default weights: λ_move=1.0, λ_fmt=0.5, λ_legal=0.5

## Hardware

Tested on a single NVIDIA B200 (192GB HBM3e) on RunPod. An 8B model with LoRA rank 64 in bf16 uses ~20GB VRAM during inference, ~60GB during training.

## References

- [Chess-R1 paper](https://arxiv.org/abs/2507.00726) (Hwang et al., 2025)
- [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Lichess puzzle database](https://database.lichess.org)

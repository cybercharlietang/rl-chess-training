# Chess-GRPO

Training LLMs to play chess via GRPO (Group Relative Policy Optimization) on Lichess puzzles. Replicates [Chess-R1](https://arxiv.org/abs/2507.00726) using TRL's `GRPOTrainer`.

## What this repo does

Given a chess puzzle (FEN position + list of legal moves), ask an LLM to produce the best move inside a `<think>…</think>` chain-of-thought followed by the move. We evaluate accuracy against Lichess' solution and use **GRPO** to fine-tune the LoRA adapter so the model picks better moves.

The repo has three interlocking pieces, each with a distinct purpose:

1. **Baseline evaluation** — measures end-to-end puzzle accuracy of a model *as it stands* (zero-shot or post-training). Answers: "on 100 held-out puzzles, how often does the model pick the correct move, how often is the move at least legal, and how often does it finish thinking inside the token budget?"
2. **Diagnostic suite** — five targeted micro-tests that isolate *what* the model understands about chess, decoupled from puzzle-solving. Answers: "does it know chess rules? can it decode a FEN? can it enumerate legal moves? can it track what happens after a move?" A model with strong declarative knowledge but weak FEN parsing will have low puzzle accuracy regardless of GRPO; the diagnostics tell us whether GRPO stands a chance of helping.
3. **GRPO training** — reinforcement-learning fine-tune using Stockfish evaluations as a dense reward signal, with LoRA on the base model.

Typical workflow: **baseline → diagnostics → train → baseline + diagnostics again, compare**.

---

## Current baselines (100 puzzles, greedy, same eval set)

| Model | Tokens | Accuracy | Legal rate | Finished thinking | Truncated |
|-------|-------:|---------:|-----------:|------------------:|----------:|
| DeepSeek-R1-Distill-Qwen-14B | 8192 | **16%** | **73%** | 80% | 20% |
| Qwen3-4B (thinking ON) | 8192 | 10% | 28% | 17% | 83% |
| Qwen3-4B (thinking ON) | 16384 | 12% | 43% | 36% | 64% |

Diagnostic scores (same N per test across models):

| Test | 14B | Qwen3-4B | Random |
|------|----:|---------:|-------:|
| FEN → Piece ID (N=25) | 76% | 24% | 7.7% |
| Legal Move Generation F1 (N=15) | 66% | 19% | 0% |
| Move Legality Judgment (N=25) | 56% | 48% | 50% |
| One-Move Consequence (N=15) | 87% | 60% | 7.7% |
| Declarative Rules (N=50) | 88% | 86% | 0% |

**Interpretation:** Both models know chess rules declaratively, but Qwen3-4B is ~3× weaker on every procedural task (FEN parsing, legal-move enumeration). That procedural weakness causes Qwen3-4B's high truncation rate — it loops trying to re-parse the FEN and hits the token cap before emitting a move.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm
apt-get install -y stockfish
```

## Data pipeline (one-time)

```bash
python -m data.download       # Lichess puzzle DB (~1 GB CSV)
python -m data.preprocess     # → data/train.jsonl (7k), eval.jsonl (3k)
```

Each JSONL row: `{fen, legal_moves (SAN list), solution_move, puzzle_rating}`.

---

## 1. Baseline evaluation (`vllm_eval.py`)

**Purpose:** Measure a model's zero-shot (or post-training) puzzle accuracy.

**What it does:** Loads a model via vLLM, runs 100 held-out puzzles with greedy decoding (T=0), extracts the move after `</think>`, scores correctness, legality, format compliance, and truncation rate. Saves a JSONL with full completions for every sample (one stats row + N sample rows).

```bash
# Any HF model, thinking toggleable
python vllm_eval.py \
    --model Qwen/Qwen3-4B \
    --n 100 \
    --max_tokens 8192 \
    --thinking 1 \
    --out results/qwen3_4b/baseline_8k.jsonl
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `Qwen/Qwen3-4B` | HF repo or local path |
| `--n` | 100 | Number of puzzles from `data/eval.jsonl` |
| `--max_tokens` | 8192 | Max generation tokens (incl. CoT) |
| `--thinking` | 1 | 1 = `enable_thinking=True` in chat template |
| `--gpu_mem` | 0.80 | vLLM's `gpu_memory_utilization` |
| `--out` | `outputs/qwen3_4b_baseline.jsonl` | Output JSONL path |

Output metrics saved in the first line of the JSONL: `accuracy`, `legal_rate`, `format_rate` (fraction with `</think>`), `truncation_rate`, `wall_s`.

---

## 2. Diagnostic suite (`chess_diagnostics/run_diagnostics.py`)

**Purpose:** Isolate what the model knows about chess, independent of puzzle-solving. A model can be great at rules but bad at FEN, or vice versa — the diagnostic tells you *which* before you spend GPU on GRPO.

**Five tests:**

| Test | Samples | What it measures | Random baseline |
|------|--------:|------------------|----------------:|
| `fen_parsing` | 25 | "In FEN X, what piece is on square e4?" | 7.7% |
| `legal_moves` | 15 | "List all legal moves for the rook on a1" (precision/recall/F1) | 0% |
| `legality` | 25 | "Is the move Bxe5 legal in this position?" (binary) | 50% |
| `consequences` | 15 | "After move Nxe5, what piece is on e5?" | 7.7% |
| `rules_knowledge` | 50 | "Can a bishop move horizontally?" (declarative yes/no + short answer) | 0% |

```bash
DIAG_THINKING=1 DIAG_MAX_TOKENS=8192 \
  python -m chess_diagnostics.run_diagnostics \
    --model Qwen/Qwen3-4B \
    --output_dir results/diagnostics_qwen3_4b

# Single test:
python -m chess_diagnostics.run_diagnostics --model Qwen/Qwen3-4B --test fen_parsing
```

**Env vars (for `model_utils.py`, vLLM backend):**
- `DIAG_THINKING` (default `1`): enable Qwen3 thinking mode in chat template
- `DIAG_MAX_TOKENS` (default `8192`): max tokens per answer
- `DIAG_GPU_MEM` (default `0.80`): vLLM memory fraction

Writes `chess_diagnostics.json` + HTML report to `--output_dir`.

---

## 3. GRPO training (`train_grpo.py`)

**Purpose:** Fine-tune the base model with RL so it picks better moves.

**Core idea of GRPO:** For each prompt, generate `G` completions, compute a reward for each, and update the policy to favor the higher-reward completions within the group. No value network. KL regularization optional (off here via `beta=0`).

All hyperparameters live in `config.py` (`Config` dataclass):

```python
# Model
model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
use_lora: True; lora_rank: 64; lora_alpha: 128
lora_target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# GRPO
num_generations: 4              # G — completions per prompt (advantage computed within group)
max_new_tokens: 4096            # bump to 8192 with FSDP; 16384 for Qwen3 to reduce truncation
temperature: 0.7
kl_penalty_coeff: 0.0           # beta=0 skips reference model (saves 28 GB/GPU)

# Training
learning_rate: 1e-5
per_device_train_batch_size: 4  # must be divisible by num_generations
gradient_accumulation_steps: 1
num_train_epochs: 3
bf16: True
gradient_checkpointing: True

# Reward — dense Stockfish + legality
reward_mode: "dense"            # or "sparse"
stockfish_depth: 15
lambda_move: 1.0                # weight on dense Stockfish reward
lambda_legal: 0.5               # weight on legality bonus

# Data
num_train_samples: 7000
num_eval_samples: 3000
puzzle_rating_min: 200          # filter puzzles by Elo
puzzle_rating_max: 2800         # (use <=1200 for easy-curriculum runs)

# Output
save_steps: 10                  # save LoRA adapter every N steps
logging_steps: 1
```

**Run:**

```bash
# Dense Stockfish rewards (recommended)
python train_grpo.py --reward_mode dense

# Short smoke run
python train_grpo.py --reward_mode dense --max_steps 16

# Resume from an adapter
python train_grpo.py --reward_mode dense --resume_adapter outputs/grpo_run/checkpoint-20
```

**Multi-GPU — FSDP, NOT DDP:** The 14B at 8192 tokens requires sharded training. DDP replicates the full 28 GB model on every GPU and OOMs on 80 GB H100 during backward. Use FSDP:

```bash
accelerate launch --config_file configs/fsdp.yaml train_grpo.py --reward_mode dense
```

(See `LESSONS.md` for the full memory math.)

**Reward breakdown:**
- `dense_stockfish_reward`: apply predicted move → run Stockfish depth 15 → sigmoid-normalize centipawn score → [0, 1]. Illegal/nonsense moves return −1.
- `is_legal_move`: binary bonus (0 or 1) for a legal move.
- Total: `λ_move · dense + λ_legal · legal` (default 1.0 + 0.5).

---

## 4. HTML reports (`build_reports.py`)

```bash
python build_reports.py
# Generates report_*.html from results/*.jsonl

python3 -m http.server 8888 --directory /workspace/rl-chess-training
# Open https://{POD_ID}-8888.proxy.runpod.net/report_<name>.html
```

Each report: summary stats, accuracy by rating bucket, per-sample chess board SVG (solution move arrow, board flipped to side-to-move perspective), predicted vs solution, correct/wrong and finished/truncated badges, and the full (never truncated) chain-of-thought expandable per sample.

---

## Architecture

```
├── README.md                    # this file
├── BACKGROUND.md                # progress log, model selection history
├── LESSONS.md                   # mistakes made, don't repeat (VRAM math, FSDP, etc.)
├── CLAUDE.md                    # working-style notes for the AI assistant
│
├── config.py                    # all hyperparameters
├── prompts.py                   # chat prompt template
├── vllm_eval.py                 # baseline evaluation (vLLM)
├── train_grpo.py                # GRPO training (TRL + LoRA)
├── build_reports.py             # HTML report generator
│
├── data/
│   ├── download.py              # Lichess puzzle DB download
│   └── preprocess.py            # → train.jsonl / eval.jsonl
├── rewards/
│   ├── sparse.py                # binary correct/incorrect
│   ├── dense_stockfish.py       # Stockfish centipawn → sigmoid
│   └── format_reward.py         # tag compliance + legal-move check + extractor
├── chess_diagnostics/           # 5-test understanding suite (vLLM backend)
├── tests/                       # unit tests (rewards, data, prompts)
└── results/
    ├── 14b/                     # DeepSeek-R1-Distill-Qwen-14B baselines
    ├── qwen3_4b/                # Qwen3-4B baselines
    └── diagnostics_*/           # per-model diagnostic scores
```

## Hardware

- **Single B200 (192 GB):** fits 14B + LoRA + 8192 tokens with DDP-equivalent
- **8× H100 SXM (80 GB each):** requires FSDP for 14B training; DDP OOMs
- **Single H100:** fine for 4B models (~12 GB weights)

## References

- [Chess-R1 paper](https://arxiv.org/abs/2507.00726) (Hwang et al., 2025)
- [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Lichess puzzle database](https://database.lichess.org)
- [Stockfish](https://stockfishchess.org)

# Chess-GRPO: Training LLMs to Play Chess via GRPO

## Goal

Build a clean, self-contained codebase that trains an LLM to play chess using GRPO (Group Relative Policy Optimization) on Lichess puzzles. This replicates the Chess-R1 paper (Hwang et al., 2025).

The codebase should be suitable as a standalone GitHub project demonstrating RL for chess.

## Background

Chess-R1 trained Qwen2.5 and Llama3.1 models with GRPO on chess puzzles. Key findings:
- Dense rewards (continuous move quality scores) >> sparse rewards (binary correct/incorrect)
- All models plateau at ~25-30% puzzle accuracy
- RL amplifies pretraining knowledge but doesn't create new chess understanding

We replicate this using **DeepSeek-R1-Distill-Qwen-7B** as the base model and **TRL's GRPOTrainer** (not veRL, which Chess-R1 used).

### Model Selection

We evaluated three models on 100 Lichess puzzles before choosing:

| Model | Accuracy | Legal Move Rate | Notes |
|-------|----------|----------------|-------|
| Qwen3-8B (thinking OFF) | 12% | 23% | Can't parse FEN reliably, no format compliance |
| Qwen3-8B (thinking ON) | 0% | 0% | Burns all tokens trying to parse FEN, never answers |
| **DeepSeek-R1-Distill-Qwen-7B** | **6%** | **66%** | Produces `<think>`/`<answer>` tags natively, strong FEN understanding |

**DeepSeek-R1-Distill-Qwen-7B** was chosen because:
1. It already uses `<think>`/`<answer>` tags natively — no need to teach format from scratch
2. 66% legal move rate shows it understands board positions far better than Qwen3
3. When it finishes reasoning (69% of samples), it produces well-structured output
4. The 6% accuracy (vs 12% for Qwen3) is because many samples truncate at 4096 tokens mid-reasoning

## Hardware

Single NVIDIA B200 (192GB HBM3e) on RunPod. Everything must run on one GPU.

## Architecture Overview

```
chess-grpo/
├── README.md                  # Project overview, results, how to reproduce
├── requirements.txt           # Pin all versions
├── config.py                  # All hyperparameters in one place
├── data/
│   ├── download.py            # Download Lichess puzzle CSV
│   └── preprocess.py          # Convert puzzles → (FEN, legal_moves, solution) training pairs
├── rewards/
│   ├── sparse.py              # Binary reward: 1 if move matches solution, 0 otherwise
│   ├── dense_stockfish.py     # Stockfish centipawn eval → normalized [0,1] reward
│   └── format_reward.py       # Format compliance reward (think/answer tags, English)
├── train_grpo.py              # Main GRPO training script using TRL
├── evaluate.py                # Evaluate model on held-out puzzle set
├── prompts.py                 # Prompt templates for Qwen3
└── scripts/
    ├── run_sparse.sh          # Train with sparse rewards
    ├── run_dense.sh           # Train with dense (Stockfish) rewards
    └── run_eval.sh            # Run evaluation
```

## Detailed Specifications

### 1. Data Pipeline (`data/`)

**Source:** Lichess puzzle database
- Download: `https://database.lichess.org/lichess_db_puzzle.csv.zst`
- This is a ~300MB compressed CSV with columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags

**Preprocessing:**
- Each puzzle has a FEN (starting position) and a Moves field (space-separated UCI moves). The first move in Moves is the opponent's move that sets up the puzzle. The remaining moves are the solution.
- Apply the opponent's first move to get the actual puzzle position
- Decompose multi-move solutions into individual (position, best_move) pairs. E.g., a puzzle with solution moves "e2e4 e7e5 d2d4" after the opponent's setup move becomes 3 training samples (but only 2 are the player's moves — alternating turns)
- For each training sample, compute the set of legal moves using `python-chess`
- Convert UCI moves to SAN notation (e.g., "e2e4" → "e4") since SAN works better than UCI per Chess-R1's findings
- Filter puzzles by rating: use 200-2800 Elo range
- Split: use the same 19.2k training samples as Chess-R1 if possible, or sample ~20k. Hold out 10k for evaluation.
- Store as JSONL with fields: `fen`, `legal_moves` (list of SAN strings), `solution_move` (SAN string), `puzzle_rating`

### 2. Prompt Template (`prompts.py`)

Follow Chess-R1's best-performing format. The prompt should include:
- System/context: "You are a professional chess player"
- The FEN string for the current position
- The complete list of legal moves in SAN notation
- A brief reminder of chess piece movement rules
- Instruction to produce reasoning in `<think>` tags and the answer in `<answer>` tags
- The answer must be a single move in SAN notation

**Important for Qwen3:** Qwen3 has a native thinking mode. We want to DISABLE the native `/think` mode and instead use the explicit `<think>`/`<answer>` tag format from Chess-R1, so the model learns to reason through GRPO rather than using its built-in reasoning. Set `enable_thinking=False` or use the non-thinking chat template.

Example prompt (adapt to Qwen3's chat format):
```
A conversation between User and Assistant. The Assistant is a professional chess player who first thinks about the reasoning process and then provides the answer.

The reasoning must be in <think> </think> tags. The answer must be in <answer> </answer> tags.

The answer must be a single move in SAN notation from the legal moves list.

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

User: The current FEN is {fen} and legal moves are {legal_moves}. What is the best move?
```

### 3. Reward Functions (`rewards/`)

**Sparse reward (`sparse.py`):**
```python
def sparse_reward(predicted_move: str, solution_move: str) -> float:
    return 1.0 if predicted_move == solution_move else 0.0
```

**Dense Stockfish reward (`dense_stockfish.py`):**
- Use `python-chess` + Stockfish binary to evaluate positions
- Given state `s` (FEN) and predicted move `a`:
  1. Apply move `a` to get new position `s'`
  2. Run `stockfish.analyse(board, chess.engine.Limit(depth=15))` on `s'`
  3. Extract centipawn score from White's perspective
  4. Normalize to [0, 1] using sigmoid: `reward = 1 / (1 + exp(-cp_score / 400))`
  5. If it's Black's turn to evaluate, flip the sign before sigmoid
- Handle mate scores: mate in N → cp_score = 10000 - N; mated in N → cp_score = -(10000 - N)
- If the predicted move is illegal (not in legal moves list), return -1.0
- Stockfish depth 15 is a good tradeoff between speed and accuracy. Each eval takes ~10-50ms.

**Format reward (`format_reward.py`):**
- Check that output contains both `<think>` and `<answer>` tags: reward 1.0 if yes, 0.0 if no
- Check that the text is in English (use a simple heuristic or `langdetect`): reward 1.0 if yes, 0.0 if no
- Check that the move inside `<answer>` tags is a legal move in SAN notation: reward 1.0 if yes, 0.0 if no

**Combined reward:**
```python
total_reward = lambda_move * move_reward + lambda_fmt * format_reward + lambda_legal * legal_reward
```
Default weights: `lambda_move=1.0, lambda_fmt=0.5, lambda_legal=0.5`

For the sparse setting: `move_reward = sparse_reward`
For the dense setting: `move_reward = dense_stockfish_reward`

### 4. GRPO Training (`train_grpo.py`)

Use **HuggingFace TRL's `GRPOTrainer`** (from `trl` package). This handles the core GRPO loop:
1. Sample G completions per prompt
2. Score with reward function
3. Compute group-normalized advantages
4. Clipped policy gradient update with KL penalty

**Key hyperparameters (put in `config.py`):**

```python
# Model
model_name = "Qwen/Qwen3-8B"
use_lora = True
lora_rank = 64
lora_alpha = 128
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO
num_generations = 16          # G: number of completions per prompt
max_new_tokens = 2048         # max length of generated response
temperature = 0.7             # sampling temperature for generation
clip_range = 0.2              # epsilon for PPO-style clipping
kl_penalty_coeff = 0.04       # beta for KL divergence penalty

# Training
learning_rate = 1e-5
batch_size = 128              # number of prompts per batch (total samples = batch_size * G)
gradient_accumulation_steps = 4
num_train_epochs = 3
warmup_ratio = 0.05
bf16 = True
gradient_checkpointing = True

# Reward
reward_mode = "dense"         # "sparse" or "dense"
stockfish_depth = 15
stockfish_path = "/usr/games/stockfish"  # install via apt

# Data
train_data_path = "data/train.jsonl"
eval_data_path = "data/eval.jsonl"
num_train_samples = 20000
num_eval_samples = 10000
```

**Training script structure:**

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

# 1. Load model + tokenizer
# 2. Apply LoRA
# 3. Define reward function that:
#    a. Parses <answer> tags to extract predicted move
#    b. Computes move reward (sparse or dense)
#    c. Computes format reward
#    d. Returns combined reward
# 4. Load training data as list of prompts
# 5. Initialize GRPOTrainer with model, reward_fn, config
# 6. trainer.train()
# 7. Save model + LoRA adapter
```

**Important TRL-specific notes:**
- TRL's GRPOTrainer expects a reward function that takes a list of completions and returns a list of float rewards
- The reward function receives the full generated text (including the prompt). You need to extract only the generated portion.
- Make sure to handle the case where the model doesn't produce valid `<answer>` tags — return a negative reward (-1.0)
- Set `num_generations` in the GRPOConfig, not in the generate call
- Check TRL's latest API — it has changed across versions. Use `trl >= 0.14.0` if available.

### 5. Evaluation (`evaluate.py`)

**Puzzle accuracy metric:**
- For each eval puzzle, generate one completion (greedy decoding, temperature=0)
- Parse the `<answer>` tag to extract the predicted move
- Check if predicted move matches the solution move (exact string match in SAN)
- Report: overall accuracy, accuracy by puzzle rating bucket (200-800, 800-1200, 1200-1600, 1600-2000, 2000-2400, 2400-2800)

**Additional metrics to track:**
- Legal move rate: % of outputs where the model produces a legal chess move
- Format compliance rate: % of outputs with correct `<think>`/`<answer>` tags
- Average reward during training (log to tensorboard/wandb)

### 6. Dependencies (`requirements.txt`)

```
torch>=2.4.0
transformers>=4.51.0  # needed for Qwen3
trl>=0.14.0
peft>=0.14.0
accelerate>=1.2.0
bitsandbytes>=0.45.0
datasets>=3.0.0
python-chess>=1.999
stockfish  # python wrapper, also need stockfish binary
wandb
langdetect
```

Also install Stockfish binary: `apt-get install stockfish`

### 7. What to run first

1. `python data/download.py` — get Lichess puzzles
2. `python data/preprocess.py` — create train.jsonl and eval.jsonl
3. `python evaluate.py --model Qwen/Qwen3-8B` — baseline eval (zero-shot, no training) to see starting accuracy
4. `python train_grpo.py --reward_mode sparse` — train with sparse rewards (expect this to mostly fail, as Chess-R1 found)
5. `python train_grpo.py --reward_mode dense` — train with dense Stockfish rewards (expect ~25-30% accuracy)
6. `python evaluate.py --model outputs/dense_checkpoint/` — evaluate trained model

### 8. Logging and checkpointing

- Log to Weights & Biases (wandb): training reward, KL divergence, puzzle accuracy at checkpoints
- Save LoRA adapter every 500 steps
- Save full evaluation results as JSON after training

## Phase 1 Status (COMPLETE)

The local skeleton is done. All non-GPU code is implemented and tested:
- Data pipeline: `data/download.py`, `data/preprocess.py` (reservoir sampling, ~30s for 5.8M rows)
- Rewards: `rewards/sparse.py`, `rewards/dense_stockfish.py`, `rewards/format_reward.py`
- Prompts: `prompts.py` (Qwen3 chat format, `<think>`/`<answer>` tags)
- Eval: `evaluate.py` (scoring logic done, inference stubbed)
- Training: `train_grpo.py` (reward wiring done, model/trainer stubbed)
- Visualizer: `visualizer.py` (Streamlit: data explorer, model outputs, GRPO training)
- Config: `config.py` (all hyperparams as dataclass)
- Tests: `tests/` (29 tests covering preprocess, rewards, prompts)

## Phase 2 Status (IN PROGRESS)

Running on RunPod B200 (192GB HBM3e). Steps 1-3 are complete, step 4 is blocked on HuggingFace auth.

### Step 1: Environment setup — DONE
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
apt-get install stockfish
```
Installed versions: torch 2.10.0 (CUDA 12.8), transformers 5.3.0, trl 0.29.0, peft 0.18.1, python-chess 1.11.2, Stockfish 16 at `/usr/games/stockfish`. Also installed pytest (not in requirements.txt).

### Step 2: Download and preprocess data — DONE
```bash
python -m data.download      # 1GB CSV, 5.8M puzzles
python -m data.preprocess    # 7k train / 3k eval
```
- Updated `config.py`: `num_train_samples=7000, num_eval_samples=3000`
- Fixed `data/preprocess.py` `__main__` to read sample counts from `Config` instead of hardcoded defaults
- Data files generated: `data/train.jsonl` (7000 samples), `data/eval.jsonl` (3000 samples)
- Rating distribution is well-spread across 400-2800 Elo, avg ~28 legal moves per position
- Note: `data/lichess_puzzles.csv` (1GB) and `data/*.jsonl` are not committed — regenerate with the above commands

### Step 3: Test Stockfish reward — DONE
**Bug found and fixed in `rewards/dense_stockfish.py`:**
- Line 57: mate score handling used `mate_in > 0` but `mate_in = 0` means checkmate already delivered (Stockfish returns `Mate(+0)`). This was incorrectly treated as "getting mated" (reward ≈ 0) instead of "delivered mate" (reward ≈ 1). Fixed to `mate_in >= 0`.

Added 10 new tests in `tests/test_rewards.py` for `dense_stockfish_reward()`:
- Illegal/nonsense moves → -1.0
- Legal moves → reward in [0, 1]
- Move quality ordering (e4 > h4 from starting position)
- Black perspective flipping (e5 > h5 after 1. e4)
- Mate-in-one → reward > 0.99
- Winning positions → high reward
- Sigmoid normalization math

**All 39 tests pass** (29 original + 10 new Stockfish tests).

### Step 4: Baseline evaluation — DONE
Evaluated three models on 100 samples to select base model (see Model Selection above).
Full 3000-sample Qwen3-8B baseline: 8.9% accuracy, 26.3% legal move rate, 0% format compliance.
Switched to DeepSeek-R1-Distill-Qwen-7B based on results.

### Step 5: Wire up training script — DONE
`train_grpo.py` fully implemented for TRL 0.29.0:
- Three separate reward functions (move, format, legal) with `reward_weights`
- Dataset as HF Dataset with chat message prompts + metadata columns
- LoRA via peft, GRPOConfig, GRPOTrainer
- Periodic eval and checkpoint saving

Fixed `format_reward.py`: `has_valid_tags()` now accepts completions where the opening `<think>` tag is part of the generation prompt (stripped during decoding), as long as `</think>` and `<answer>` are present.

### Step 6: GRPO training with dense rewards — DONE (16-step pilot)
```bash
python train_grpo.py --reward_mode dense --max_steps 16
```
Config: batch_size=8, num_generations=8, LoRA rank 64, lr=1e-5, dense Stockfish rewards.
~85s/step on B200, 16 steps completed in 23 minutes.

**Training metrics (step 10 → step 16):**
| Metric | Step 10 | Step 16 |
|--------|---------|---------|
| Total reward | 0.843 | 1.202 |
| Move reward (dense) | -0.323 | -0.200 |
| Format reward | 0.667 | 0.778 |
| Legal move rate | 0.500 | 0.625 |

**Eval results (100 samples, greedy decoding):**
| Metric | Baseline | After 16 steps |
|--------|----------|---------------|
| Accuracy | 6% | 9% (+3%) |
| Legal move rate | 66% | 53% (-13%) |
| Format compliance | 69% | 61% (-8%) |
| Finished reasoning | 69% | 65% (-4%) |
| Avg completion length | 11K chars | 37K chars |

**Key finding — truncation is the main bottleneck:**
- ~30% of completions truncate at the 4096 token limit because the model gets stuck in FEN parsing loops ("Wait, no... Wait, maybe...")
- 29/31 truncated baseline samples show this "Wait" loop pattern
- GRPO fixed 20 previously-truncated samples but broke 24 previously-working ones (net: slightly worse)
- Truncated completions get garbage rewards (no `<answer>` tag → move reward = -1.0)
- For GRPO to work well, this truncation rate needs to come down

**Implications for next run:**
- Consider a larger model (14B) that may parse FEN more reliably
- Or reduce max_completion_length to penalize verbose reasoning
- Or add a length penalty to the reward function
- Longer training (200+ steps) may help the model learn to avoid looping

### Step 7: 14B Model Evaluation — COMPLETED

Evaluated `DeepSeek-R1-Distill-Qwen-14B` as a potential upgrade from the 7B.

**Baseline comparison (100 puzzles, 4096 tokens):**

| Metric | 7B Baseline | 14B Baseline |
|--------|------------|-------------|
| Accuracy | 6% | 5% |
| Legal Move Rate | 66% | 51% |
| Format Compliance | 69% | 51% |
| Finished Reasoning | 69% | 62% |

**14B is worse on every metric.** The larger model's extended reasoning produces more FEN parsing loops, not better chess play.

**10-step GRPO pilot on 14B (batch=16, dense rewards):**
- Reward improved slightly but overall worse than the 7B pilot
- VRAM: 181.7 / 183.4 GB — nearly maxed out at batch_size=16

**Conclusion:** Do not scale within the DeepSeek-R1-Distill family. Go back to 7B.

### Step 8: Diagnostic Test Suite — COMPLETED

Built a 5-test diagnostic suite (`chess_diagnostics/`) to measure what the 14B model actually understands about chess. Results (4096 tokens, color-explicit prompts, strict scoring):

| Test | Score | Baseline | Interpretation |
|------|-------|----------|---------------|
| Declarative Rules Knowledge | **88%** | 0% | Knows chess rules in theory |
| FEN → Piece Identification | **76%** | 7.7% | Can decode FEN notation |
| Legal Move Generation | **66% F1** | 0% | Can enumerate moves for a piece |
| Move Legality Judgment | **56%** | 50% | Near-random at verifying legality |
| One-Move Consequence | **87%** | 7.7% | Can track board state after a move |

**Key findings:**
1. The model has strong declarative knowledge (88%) but a gap between knowing rules and applying them compositionally.
2. FEN parsing works (76%) but requires extensive chain-of-thought (~1000+ tokens for simple positions).
3. It can enumerate legal moves (66% F1) and track consequences (87%) — procedural tasks amenable to step-by-step reasoning.
4. It cannot judge move legality (56%, near random) — this requires compositional reasoning about blocking, pins, and checks.
5. Color disambiguation in prompts is critical: saying "it is black's turn, after black plays Kg8" vs just "after Kg8 is played" improved consequence tracking from 40% → 87%.

**Implications for GRPO:** The model has a foundation to build on — it's not pure random guessing. GRPO should focus on:
- Reducing FEN parsing verbosity (the main source of truncation)
- Improving move selection quality (the model can generate moves but picks poorly)
- The 14B model will be used with an 8192 token limit to reduce truncation

### Step 9: Longer GRPO training on 14B — TODO
Full training run on the 14B model with 8192 token limit. The hypothesis is that doubling the token budget (from 4096 to 8192) will reduce truncation enough for the 14B's stronger reasoning to outperform the 7B. No length penalty or max completion length reduction for now — let the model use the full context.

## Dashboard & Visualization Preferences

**Serving method:** Static HTML served on port 8888 via RunPod proxy. Do NOT use Streamlit — it doesn't work through RunPod's proxy.
- URL pattern: `https://{RUNPOD_POD_ID}-8888.proxy.runpod.net/comparison.html`
- Get pod ID from `$RUNPOD_POD_ID` env var
- Serve with: `python3 -m http.server 8888 --directory /root/rl-chess-training`
- Generate reports with: `python generate_html_report.py`

**Dashboard format for evaluation results:**
- **Summary Statistics table:** Accuracy, Legal Move Rate, Format Compliance, Finished Reasoning, Avg Completion Length
- **Accuracy by Rating Bucket table**
- **Truncation Cross-tab** (when comparing before/after GRPO): 2x2 matrix of Baseline Finished/Truncated vs Trained Finished/Truncated
- **Training Curve** (during GRPO): Total Reward, Move Reward, Format Reward, Legal Move Rate at each logged step, with a graph
- **Side-by-Side Sample Browser:** Per-sample list with rating, solution, status (CORRECT/WRONG), truncation (FINISHED/TRUNCATED), expandable completions
- **Chess board rendering:** Each sample should show the board position (SVG) with the solution move highlighted (green arrow), oriented from the side to move's perspective
- **Full model output:** Never truncate completions in the report — show the entire model output for each sample
- **Sample detail view:** Board SVG + FEN + solution + predicted move + rating, followed by the full model completion text

## References

- Chess-R1 paper: https://arxiv.org/abs/2507.00726
- Chess-R1 code: https://github.com/krafton-ai/Chess-R1
- TRL GRPO docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- DeepSeek-R1: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- Lichess puzzles: https://database.lichess.org
- Stockfish: https://stockfishchess.org
# Chess-GRPO

Training LLMs to play chess via GRPO (Group Relative Policy Optimization) on Lichess puzzles, replicating [Chess-R1](https://arxiv.org/abs/2507.00726) using TRL's `GRPOTrainer`.

## Current baselines (100 puzzles, greedy, same eval set)

| Model | Tokens | Accuracy | Legal rate | Finished thinking | Truncated |
|-------|-------:|---------:|-----------:|------------------:|----------:|
| DeepSeek-R1-Distill-Qwen-14B | 8192 | **16%** | **73%** | 80% | 20% |
| Qwen3-4B (thinking ON) | 8192 | 10% | 28% | 17% | 83% |
| Qwen3-4B (thinking ON) | 16384 | 12% | 43% | 36% | 64% |

**Diagnostic (5-test suite):** The 14B beats Qwen3-4B on every procedural task (FEN parsing 76% vs 24%, legal-move enumeration 66% vs 19% F1, consequence tracking 87% vs 60%) but both are equally good on declarative rules (88% vs 86%). See `results/diagnostics_*/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm
apt-get install -y stockfish
```

## Usage

### 1. Prepare data (one-time)
```bash
python -m data.download       # Downloads Lichess puzzle DB (~1GB)
python -m data.preprocess     # Writes data/train.jsonl + eval.jsonl
```

### 2. Baseline evaluation (vLLM, fast)
```bash
# Any HF model, with or without thinking mode
python vllm_eval.py --model Qwen/Qwen3-4B --n 100 --max_tokens 8192 --thinking 1 \
    --out results/qwen3_4b/baseline_8k.jsonl

python vllm_eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --n 100 --max_tokens 8192 \
    --thinking 0 --out results/14b/baseline_8k.jsonl
```

### 3. Diagnostic suite (FEN parsing, legal moves, legality, consequences, rules)
```bash
DIAG_THINKING=1 DIAG_MAX_TOKENS=8192 \
  python -m chess_diagnostics.run_diagnostics --model Qwen/Qwen3-4B \
    --output_dir results/diagnostics_qwen3_4b
```

### 4. GRPO training
```bash
python train_grpo.py --reward_mode dense
# Short test run:
python train_grpo.py --reward_mode dense --max_steps 16
```

### 5. HTML reports
```bash
python build_reports.py  # reads results/*.jsonl, writes reports/*.html
python3 -m http.server 8888
# Open https://{POD_ID}-8888.proxy.runpod.net/reports/<file>.html
```

## Architecture

```
├── config.py                    # Hyperparameters
├── prompts.py                   # Chat prompt template
├── vllm_eval.py                 # Fast vLLM baseline evaluation
├── train_grpo.py                # GRPO training (TRL + LoRA)
├── build_reports.py             # HTML report generator
├── data/
│   ├── download.py              # Download Lichess puzzle DB
│   └── preprocess.py            # JSONL: FEN, legal_moves, solution
├── rewards/
│   ├── sparse.py                # Binary reward
│   ├── dense_stockfish.py       # Stockfish-normalized dense reward
│   └── format_reward.py         # Tag compliance + legal-move check
├── chess_diagnostics/           # 5-test understanding suite
└── results/
    ├── 14b/, qwen3_4b/          # Baseline JSONLs per model
    └── diagnostics_*/           # Per-model diagnostic scores
```

## Reward

`reward = λ_move · dense_stockfish + λ_legal · legal` with `λ_move=1.0`, `λ_legal=0.5`.
Dense move reward applies predicted move, runs Stockfish depth 15, normalizes via `sigmoid(cp/400)`.

## Hardware

Tested on 1× B200 (192GB) and 8× H100 SXM (80GB each). The 14B+LoRA+8192 tokens fits a single B200, or 8×H100 via FSDP (DDP OOMs — see `LESSONS.md`).

## References
- [Chess-R1 paper](https://arxiv.org/abs/2507.00726) (Hwang et al., 2025)
- [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Lichess puzzles](https://database.lichess.org)

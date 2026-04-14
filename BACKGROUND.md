# Chess-GRPO: Background & Progress

## Goal

Train an LLM to play chess via GRPO (Group Relative Policy Optimization) on Lichess puzzles. Replicates [Chess-R1](https://arxiv.org/abs/2507.00726) (Hwang et al., 2025) using TRL's `GRPOTrainer` instead of veRL.

## Key findings from Chess-R1

- Dense rewards (continuous move-quality scores) ≫ sparse rewards (binary correct/incorrect)
- All models plateau at ~25–30% puzzle accuracy
- RL amplifies pretraining knowledge but doesn't create new chess understanding

## Model selection

Zero-shot baselines on 100 puzzles, greedy, same eval set, same extractor:

| Model | Tokens | Accuracy | Legal | Finished | Notes |
|-------|-------:|---------:|------:|---------:|-------|
| **DeepSeek-R1-Distill-Qwen-14B** | 8192 | **16%** | **73%** | 80% | Current base model |
| DeepSeek-R1-Distill-Qwen-7B | 4096 | 6% | 66% | 69% | Earlier base; truncates at 4K |
| Qwen3-4B (thinking ON) | 16384 | 12% | 43% | 36% | FEN parse loop truncates 64% |
| Qwen3-4B (thinking ON) | 8192 | 10% | 28% | 17% | FEN parse loop truncates 83% |
| Qwen3-8B (thinking OFF) | 4096 | 12% | 23% | 0% | No format compliance |
| Qwen3-8B (thinking ON) | 4096 | 0% | 0% | 0% | Burns tokens on FEN parsing |

**Current default:** `DeepSeek-R1-Distill-Qwen-14B`. FEN parsing is the key bottleneck — see diagnostics below.

## Diagnostic suite results (same N per test across models)

| Test | 14B | Qwen3-4B | Random |
|------|----:|---------:|-------:|
| FEN → Piece Identification (N=25) | 76% | **24%** | 7.7% |
| Legal Move Generation F1 (N=15) | 66% | **19%** | 0% |
| Move Legality Judgment (N=25) | 56% | **48%** | 50% |
| One-Move Consequence (N=15) | 87% | **60%** | 7.7% |
| Declarative Rules Knowledge (N=50) | 88% | **86%** | 0% |

Both models have near-identical **declarative** knowledge (86 vs 88%) but Qwen3-4B is ~3× weaker on every **procedural** task. This is the root cause of Qwen3-4B's high truncation rate: it can't reliably decode the FEN, loops trying, hits the token cap. Data in `results/diagnostics_*/`.

## Current approach (Step 10 — 2026-04-05/06)

After the Step 10 redesign, the production setup is:

- **Prompts:** Native `<think>…</think>` reasoning (no custom `<answer>` tag). Move extracted from text after `</think>` via a priority parser: bold markdown → "is <SAN>" pattern → standalone SAN line → first SAN token.
- **Reward:** Dense Stockfish (depth 15, sigmoid-normalized cp) + legality. No format/language reward — format is already implicit in legal-move scoring.
- **Eval:** vLLM (not HF `.generate`) — 5–10× faster. `vllm_eval.py` loads any HF model with any `enable_thinking` setting.
- **GRPO config:** G=4, per-device batch=4, grad-accum=1, 8192 tokens, `beta=0` (no reference model — saves 28GB/GPU).
- **Hardware:** 8×H100 SXM. Training **requires FSDP** — DDP OOMs for 14B at 8192 tokens (see `LESSONS.md`).

## Step 11 — Qwen3-4B experiment (2026-04-14)

Ran Qwen3-4B (thinking=True) through the full baseline + diagnostic pipeline on 8×H100 SXM.

**Baseline (100 puzzles, same eval set, greedy via vLLM):**
- 8K tokens: 10% / 28% legal / 17% finished / 83% truncated
- 16K tokens: 12% / 43% legal / 36% finished / 64% truncated

**Diagnostic:** Declarative knowledge strong (86%) but procedural tasks weak — FEN parsing at 24% vs 14B's 76% explains the loop truncation.

**Conclusion:** Qwen3-4B is not competitive with the 14B as a GRPO base on this task. The FEN parsing weakness is structural (not addressable by more tokens), so GRPO would get mostly garbage rollouts (64% truncation → no `</think>` → no move → -1 reward). Do not train on 4B unless we switch to a model with stronger procedural grounding (candidates: `Qwen3-4B-Thinking-2507`, `DeepSeek-R1-Distill-Qwen-1.5B`, `Phi-4-mini-reasoning`).

Data: `results/qwen3_4b/baseline_{8,16}k.jsonl`, `results/diagnostics_qwen3_4b/`.

## Historical progression (obsolete but referenced by LESSONS.md)

- **Steps 1–5 (2026-03-13/17):** Data pipeline, Stockfish reward, 7B training. Key bug: mate-score normalization sign error. 39 unit tests.
- **Step 6 (2026-03-17):** 16-step GRPO pilot on 7B @ 4096 tokens. Reward 0.84→1.20. Net accuracy +3%, but legal rate −13% — truncation dominated.
- **Step 7:** 14B evaluation. At 4096 tokens 14B underperformed 7B (5% vs 6% accuracy) — larger model loops more.
- **Step 8:** Built 5-test diagnostic suite. Discovered declarative vs procedural gap.
- **Step 9 (2026-03-18):** 14B @ 8192 tokens solved the truncation problem. 30 steps on easy puzzles (<1200 Elo) → format 87%, legal 78%. Best adapter in `outputs/grpo_easy_r2/final_adapter`.
- **Step 10 (2026-04-05/06):** Redesigned for 8×H100 SXM. Removed `<answer>` tags + format reward. DDP OOM'd repeatedly — must use FSDP. Baseline on new setup: 16% / 73%.

## Dashboard & serving

Static HTML over RunPod proxy (Streamlit doesn't work through the proxy):

- Build: `python build_reports.py` (reads `results/*.jsonl`)
- Serve: `python3 -m http.server 8888 --directory /workspace/rl-chess-training`
- URL: `https://{POD_ID}-8888.proxy.runpod.net/reports/<file>.html`

Each report includes: summary stats, accuracy-by-rating-bucket, per-sample board SVG (solution move arrow, oriented to side-to-move), FEN, predicted/solution, correct/wrong + finished/truncated badges, and the full (never truncated) chain-of-thought expandable per sample.

## References

- Chess-R1 paper: https://arxiv.org/abs/2507.00726
- Chess-R1 code: https://github.com/krafton-ai/Chess-R1
- TRL GRPO: https://huggingface.co/docs/trl/main/en/grpo_trainer
- DeepSeek-R1: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- Qwen3: https://huggingface.co/Qwen/Qwen3-4B
- Lichess puzzles: https://database.lichess.org
- Stockfish: https://stockfishchess.org

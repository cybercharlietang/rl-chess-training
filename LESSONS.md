# Lessons Learned

## Diagnostic Test Scoring

### Always validate your scorer against the raw model output
In the first diagnostic run, Test 1 (FEN parsing) reported 24% accuracy with lenient scoring. The scorer used substring matching (`if correct in answer`) against the last line of output. This counted mid-reasoning mentions as correct answers — e.g., the model said "d6=N (white knight)" while parsing rank 6, and the scorer matched "white knight" even though the model never produced a final answer.

**Fix:** Only score explicit final answers: text after `</think>` tags, inside `<answer>` tags, or matching a concluding pattern like "the answer is X". If the model doesn't conclude, score it as wrong.

**Result:** After fixing, FEN parsing accuracy went from 24% → 76% (with 4096 tokens). Many previously "correct" answers were false positives, but with enough tokens the model actually does answer correctly most of the time.

### Token budget determines whether you're testing the model or testing your truncation
The first diagnostic run used `max_new_tokens=512`. The model's chain-of-thought FEN parsing consumed all 512 tokens before reaching an answer in most cases. This meant we were measuring "can the model answer in 512 tokens" not "can the model answer at all."

**Fix:** Use `max_new_tokens=4096` for diagnostic tests. The model needs extensive reasoning to parse FEN notation step by step.

**Impact:** Scores changed dramatically:
- FEN parsing: 24% → 76%
- Legal move generation: 0% → 66%
- Move legality: 0% → 48%
- One-move consequence: 7% → 40%

### Move notation normalization matters
Test 2 (legal move generation) scored 0% F1 even after fixing the token budget. The model was producing correct answers — e.g., outputting "c3" for a pawn move, "a6, a5" for pawn moves — but the parser couldn't extract them because:

1. The `parse_move_list` function joined lines with spaces then tried to split on commas, producing one long string that got filtered out by length.
2. The model sometimes used coordinate notation (e.g., "h7h6") instead of SAN ("h6"), and the parser didn't normalize.
3. The model embedded moves in sentences ("The pawn can only move to c3"), and the parser only looked at standalone tokens.

**Fix:**
- Process lines from the `</think>` section individually, don't join them
- Extract move-like tokens from longer text using regex (`[KQRBN]?[a-h]?x?[a-h][1-8][+#]?`)
- Try normalizing bare square names (e.g., "c3") to SAN by checking legal moves from the piece's square

**Result:** Legal move generation F1 went from 0% → 66%.

## Model Evaluation

### Don't assume larger models are better — test first
We assumed the 14B model would outperform the 7B on chess puzzles due to more parameters. The baseline evaluation showed the opposite:

| Metric | 7B | 14B |
|--------|-----|------|
| Accuracy | 6% | 5% |
| Legal move rate | 66% | 51% |
| Format compliance | 69% | 51% |

The 14B model's extended reasoning ability (from R1 distillation) actually made it *worse* — more capacity for self-doubt loops and FEN re-parsing spirals.

### The truncation problem is behavioral, not capacity-limited
The hypothesis was that the 14B would truncate less because it has "more intelligence" to parse FEN efficiently. Instead, truncation got worse. The R1-Distill training teaches models to do extended chain-of-thought, and both the 7B and 14B learned this behavior. The 14B is just better at generating *more* reasoning, which isn't helpful when the reasoning is circular.

### VRAM estimates are unreliable — always leave headroom
My VRAM estimate for the 14B with batch_size=16 was ~145-160 GB. Actual usage was 181.7 GB out of 183.4 GB. The gap came from:
- PyTorch memory allocator fragmentation
- Activation memory being higher than estimated with gradient checkpointing
- TRL internal buffers not accounted for

**Lesson:** Estimate conservatively and start with smaller batches. Monitor `nvidia-smi` after the first step before committing to a long run.

## Diagnostic Test Design

### Test declarative knowledge separately from procedural application
The 14B model scores 88% on declarative chess rules (yes/no and short-answer questions) but only 48% on applying those rules to judge move legality (equivalent to random guessing). This is a critical distinction:

- **Declarative:** "Can a bishop move horizontally?" → "No" (88% correct)
- **Procedural:** "In this FEN position, is Bxe5 legal?" → random guessing (48%)

This gap means the model has memorized chess rules from training data but cannot compose them into multi-step reasoning about specific positions. GRPO training is trying to bridge this gap.

### Separate what you're testing from how you're testing it
The legality test (Test 3) asks "Is move X legal? Answer yes or no." The model gets 48% — random. But the legal move generation test (Test 2) asks "List all legal moves for this piece" and the model gets 66% F1. The model can enumerate moves but can't verify a specific move — likely because enumeration is a more constrained procedure (go through each direction) while verification requires checking blocking, pins, and checks compositionally.

### Simple positions reveal more than complex ones
Using positions with 3-8 pieces (endgames) rather than full board positions was the right call. It eliminates confounding variables: with few pieces, FEN parsing is easier, there are fewer legal moves, and errors are clearly attributable to rule application rather than parsing overwhelm.

### SAN notation is ambiguous without color context — the model thinks in FEN
SAN notation uses uppercase letters for all pieces regardless of color (`Kg5` means "king to g5" for whichever side is moving). But DeepSeek-R1 models think in FEN where `K` = white king and `k` = black king. When asking "is the move Kg5 legal?" without specifying whose turn it is, the model may check if the *white* king can go to g5 even when it's black's turn.

**Fix:** Always include color in questions:
- Bad: "after the move Nxc8 is played..."
- Good: "it is white's turn. After white plays the move Nxc8..."

**Impact:**
- One-Move Consequence: 40% → **87%** (more than doubled)
- Move Legality: 48% → **56%** (small improvement — this test has deeper issues)

The consequence test had the biggest gain because it just needed to know *which color's piece* ends up on the target square. The legality test barely improved because the difficulty is in compositional reasoning (blocking, pins, checks), not color identification.

**Takeaway:** When a model underperforms, check if the prompt is ambiguous *from the model's perspective*, not just from a chess player's perspective. The model's mental model of notation may differ from standard conventions.

## Infrastructure

### RunPod proxy URL format
Streamlit doesn't work through RunPod's proxy. Use static HTML served on port 8888 instead:
- URL: `https://{RUNPOD_POD_ID}-8888.proxy.runpod.net/filename.html`
- Pod ID from `$RUNPOD_POD_ID` env var
- Serve with: `python3 -m http.server 8888 --directory /root/rl-chess-training`

### Always show full model output in reports
Never truncate completions in diagnostic reports. The full chain-of-thought reveals *why* the model fails — is it looping? Misunderstanding FEN? Getting the right intermediate result but failing to conclude? Truncated output hides these patterns and leads to wrong conclusions about model capability.

### Use the actual field names from the data, not assumed ones
When writing report generators, read the data first to check field names. The diagnostic JSON uses `accuracy` (not `score`), `is_correct` (not `correct`), `raw_answer` (not `model_output`), and `correct_answer` (not `expected`). Using `.get("score", 0)` silently returned 0 for every test, producing a report with all zeros that looked plausible but was completely wrong. Always verify report output against known values before sharing.

## GRPO Training

### 8192 tokens dramatically reduces truncation for the 14B model
The 14B model at 4096 tokens had 51% format compliance and 51% legal move rate — largely because ~38% of completions truncated mid-reasoning. At 8192 tokens, avg completion length was ~3,600 tokens (well within budget) and format compliance jumped to 87%, legal move rate to 78%. The model doesn't need 8192 tokens — it needs enough headroom to finish its chain-of-thought.

### num_generations=2 works for piloting but is noisy
GRPO with only 2 completions per prompt still showed reward improvement (1.46 → 1.63 over 20 steps), but the advantage signal is inherently noisy — each step only compares "this one was better/worse than the other one." For production runs, use at least 4-8 generations. For quick pilots to check if training is working at all, 2 is sufficient.

### Easy puzzles first is a sound curriculum strategy
Training on puzzles rated <1200 (2043 samples) showed clear improvement in format compliance and legal move rate within 20 steps. Starting with easy puzzles means the model gets more positive reward signal (it can actually solve some of them), which gives GRPO better gradients to learn from. Scale up to harder puzzles once the model has learned basic formatting and move selection.

### Pre-launch checklist for GPU training runs
Before every training launch, verify:
1. **VRAM is clear** — `nvidia-smi` shows 0 MiB. Kill stale processes first.
2. **`save_steps` matches run length** — if running N steps, set `save_steps` ≤ N/2 so you never lose more than half the work on a crash or force-kill. We lost an entire 20-step overnight run because `save_steps=500` was left over from a 5000-step plan.
3. **No `tee` in launch scripts** — it buffers output and can mask errors. Use `> log 2>&1`.
4. **Plan the full run upfront** — don't chain multiple short runs that each reload the model. Each relaunch costs ~10 min of loading overhead and risks zombie processes holding VRAM. If you want 60 steps, run 60 steps, not 10+50.
5. **Test adapter loading** before committing to a long run — verify the adapter loads and training starts (step 1 completes) before walking away.

This cost us ~5 hours of wasted GPU time across multiple failed launches.

### Step time varies significantly — don't extrapolate from step 1
Step 1 took 7.4 min (includes compilation, warmup). Step 2 took 5.4 min. Later steps averaged ~10 min as the model produced longer completions during training. Always wait for 3-5 steps before estimating total runtime.

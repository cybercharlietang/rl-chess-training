# Chess LLM Diagnostic Test Suite — Claude Code Prompt

## Context

I'm running diagnostic tests on chess LLMs to measure what they actually understand about chess, independent of GRPO training. The hypothesis is that models like DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Qwen-14B lack basic chess knowledge, and their puzzle accuracy (~6%) is just random guessing from the legal move list.

I need a self-contained Python script that runs 4 diagnostic tests on a HuggingFace model and produces a clean HTML report with results.

## Environment

- RunPod with B200 GPU (192GB VRAM)
- Models already downloaded at `/workspace/models/` or can be loaded from HuggingFace
- `python-chess`, `torch`, `transformers` installed
- Stockfish installed at `/usr/games/stockfish`

## Test Design

### Test 1: FEN → Piece Identification (50 samples)

**What it tests:** Can the model decode FEN notation at the most basic level?

**Setup:**
- Generate 50 random positions from the Lichess puzzle dataset (or generate random legal positions using `python-chess`)
- For each position, pick a random occupied square
- Ask: "In the position with FEN `{fen}`, what piece is on square `{square}`? Answer with exactly one of: white king, white queen, white rook, white bishop, white knight, white pawn, black king, black queen, black rook, black bishop, black knight, black pawn, empty. Answer only with the piece name, nothing else."

**Scoring:**
- Exact match (case-insensitive, strip whitespace)
- Record: accuracy, breakdown by piece type, most common errors

**Baseline:** Random guessing from 13 options = ~7.7%. But smart guessing (always say "empty" since ~50% of squares are empty) = ~50%.

### Test 2: Legal Move Generation (30 samples)

**What it tests:** Can the model determine legal moves from a position?

**Setup:**
- 30 positions from Lichess puzzles (varied complexity — some with few legal moves, some with many)
- For each position, pick a specific piece that has legal moves
- Ask: "In the position with FEN `{fen}`, list ALL legal moves for the {piece_type} on {square}. Use SAN notation. List only the moves, separated by commas. Do not include moves for any other piece."

**Scoring:**
- Compute precision: what fraction of the model's listed moves are actually legal for that piece?
- Compute recall: what fraction of the piece's actual legal moves did the model list?
- Compute F1 from precision and recall
- Also track: how often does the model list moves for the WRONG piece, or list moves that aren't even valid SAN notation?

**Ground truth:** Use `python-chess` to compute the actual legal moves for that piece.

### Test 3: Move Legality Judgment (50 samples)

**What it tests:** Can the model judge whether a specific move is legal?

**Setup:**
- 50 positions, each with a proposed move
- 25 of the moves are legal, 25 are illegal
- Illegal moves should be plausible-looking (e.g., a bishop move that's blocked by another piece, a move into check, a pawn moving backwards) — not obviously nonsensical
- Ask: "In the position with FEN `{fen}`, is the move `{move}` legal? Answer only 'yes' or 'no'."

**Scoring:**
- Accuracy, precision, recall, F1 for the binary classification
- Breakdown: what types of illegal moves does the model miss? (blocked paths, moving into check, wrong piece movement pattern, etc.)

**Baseline:** Random = 50%. Always saying "yes" = 50%.

### Test 4: One-Move Consequence (30 samples)

**What it tests:** Can the model track what happens after a single move?

**Setup:**
- 30 positions with a specified legal move (use captures for half, non-captures for half)
- Ask: "In the position with FEN `{fen}`, after the move `{move}` is played, what piece is on square `{target_square}`? Answer with exactly one of: white king, white queen, white rook, white bishop, white knight, white pawn, black king, black queen, black rook, black bishop, black knight, black pawn, empty."
- For captures: target_square is the capture square (tests whether model knows the capturing piece replaces the captured one)
- For non-captures: target_square is the destination square (tests basic move tracking)
- Include 5 special cases: castling (ask what's on g1 after O-O), en passant (ask what's on the en passant square), promotion

**Scoring:**
- Exact match accuracy
- Breakdown: captures vs non-captures vs special moves

**Ground truth:** Use `python-chess` to apply the move and read the resulting board.

## Implementation Requirements

### Model Loading
```python
# Support loading any HuggingFace model
# Default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# Also test: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# Load in bf16, use GPU
```

### Generation Config
- Temperature = 0 (greedy decoding)
- max_new_tokens = 512 (these are short-answer questions, not puzzles)
- Use the model's chat template via tokenizer.apply_chat_template()
- System prompt: "You are a chess expert. Answer the question precisely and concisely."

### Position Generation
- Use positions from the Lichess puzzle CSV if available at `/workspace/rl-chess-training/data/`
- Otherwise, generate positions using `python-chess` by playing random games for 20-40 moves
- Ensure variety: include openings, middlegames, endgames
- For Test 3 (illegal moves), generate plausible illegal moves programmatically:
  - Take a legal move and modify it slightly (e.g., bishop move but blocked by a piece in between)
  - Move a piece to a square occupied by a friendly piece
  - Move a king into check
  - Move a pinned piece

### Output

**Console:** Print running progress and per-test summary stats.

**HTML Report:** Generate a single `chess_diagnostics.html` file with:
- Summary table: test name, accuracy, baseline, number of samples
- For each test, show every sample: the position (as ASCII board), the question, the model's answer, the correct answer, and whether it was correct
- Color-code: green for correct, red for wrong
- At the top: model name, total time, date

**JSON:** Also save raw results as `chess_diagnostics.json` for further analysis.

### File Structure
```
chess_diagnostics/
├── run_diagnostics.py      # Main script
├── test_fen_parsing.py     # Test 1
├── test_legal_moves.py     # Test 2
├── test_legality.py        # Test 3
├── test_consequences.py    # Test 4
├── position_generator.py   # Generate test positions
├── model_utils.py          # Model loading and inference
├── report.py               # HTML report generation
└── README.md               # What this is and how to run it
```

### Running
```bash
# Run all tests on default model
python run_diagnostics.py

# Run on specific model
python run_diagnostics.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# Run single test
python run_diagnostics.py --test fen_parsing

# Output goes to /workspace/rl-chess-training/results/diagnostics/
```

## Key Considerations

1. **Parsing model output:** The models may produce verbose reasoning even for simple questions. Be robust — extract the answer from wherever it appears in the output. Look for the answer after `</think>` tags if present. Use regex fallbacks.

2. **FEN positions should be diverse:** Mix of piece counts (full board vs endgame), complexity levels, and both colors to move.

3. **Reproducibility:** Set random seed (42). Save the exact positions used so results can be compared across models.

4. **Speed:** With 160 total samples and max_new_tokens=512, this should run in 15-30 minutes per model on a B200.

5. **The null hypothesis we're testing:** The model's performance on each test is no better than the specified random baseline. If it's at or below baseline on ALL tests, we can conclude the model has zero chess understanding and its puzzle accuracy is pure noise.

"""Convert Lichess puzzles into (FEN, legal_moves, solution_move) training pairs."""

import json
import os
import random

import chess

from data.download import RAW_CSV_PATH, iter_raw_puzzles

TRAIN_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval.jsonl")


def decompose_puzzle(puzzle: dict) -> list[dict]:
    """Decompose a single puzzle into individual (position, best_move) samples.

    A puzzle has:
      - FEN: starting position
      - Moves: space-separated UCI moves where moves[0] is the opponent's
        setup move and moves[1:] are the solution (alternating player/opponent)

    We take only the player's moves as training targets (indices 1, 3, 5, ...).
    """
    fen = puzzle["FEN"]
    moves_uci = puzzle["Moves"].split()
    rating = int(puzzle["Rating"])

    board = chess.Board(fen)

    # Apply the opponent's setup move
    setup_move = chess.Move.from_uci(moves_uci[0])
    if setup_move not in board.legal_moves:
        return []
    board.push(setup_move)

    samples = []
    for i, uci_str in enumerate(moves_uci[1:], start=1):
        move = chess.Move.from_uci(uci_str)

        if i % 2 == 1:
            # Player's move — this is a training target
            legal_moves_san = [board.san(m) for m in board.legal_moves]
            solution_san = board.san(move)
            samples.append({
                "fen": board.fen(),
                "legal_moves": legal_moves_san,
                "solution_move": solution_san,
                "puzzle_rating": rating,
                "puzzle_id": puzzle["PuzzleId"],
            })

        if move not in board.legal_moves:
            break
        board.push(move)

    return samples


def preprocess(
    csv_path: str = RAW_CSV_PATH,
    num_train: int = 700,
    num_eval: int = 300,
    rating_min: int = 200,
    rating_max: int = 2800,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Two-phase approach for efficiency:

    Phase 1: Reservoir-sample *puzzles* from the CSV stream (cheap — just
    dict copies, no chess logic). We over-sample by 2x since each puzzle
    yields ~1-3 training samples after decomposition.

    Phase 2: Decompose only the sampled puzzles (expensive — python-chess).
    Then take the first (num_train + num_eval) samples.

    This avoids calling decompose_puzzle() on millions of unused rows.
    """
    total_needed = num_train + num_eval
    # Over-sample puzzles: each puzzle yields ~1.5 samples on average,
    # so we need roughly total_needed puzzles. 2x buffer for safety.
    puzzle_reservoir_size = total_needed * 2
    rng = random.Random(seed)

    # Phase 1: reservoir-sample puzzles (fast — no chess logic)
    puzzle_reservoir: list[dict] = []
    seen = 0
    puzzles_scanned = 0

    print(f"Phase 1: Reservoir-sampling puzzles from {csv_path} ...", flush=True)
    print(f"  Target: ~{puzzle_reservoir_size} puzzles for {total_needed} samples", flush=True)

    for puzzle in iter_raw_puzzles(csv_path):
        puzzles_scanned += 1

        rating = int(puzzle["Rating"])
        if not (rating_min <= rating <= rating_max):
            continue

        seen += 1
        if len(puzzle_reservoir) < puzzle_reservoir_size:
            puzzle_reservoir.append(puzzle)
        else:
            j = rng.randint(0, seen - 1)
            if j < puzzle_reservoir_size:
                puzzle_reservoir[j] = puzzle

        if puzzles_scanned % 1_000_000 == 0:
            print(f"  Scanned {puzzles_scanned} puzzles...", flush=True)

    print(f"  Done. Scanned {puzzles_scanned} puzzles, "
          f"{seen} in rating range, kept {len(puzzle_reservoir)}.", flush=True)

    # Phase 2: decompose only the sampled puzzles (slow but small set)
    print(f"Phase 2: Decomposing {len(puzzle_reservoir)} puzzles...", flush=True)
    rng.shuffle(puzzle_reservoir)

    all_samples = []
    for p in puzzle_reservoir:
        all_samples.extend(decompose_puzzle(p))
        if len(all_samples) >= total_needed:
            break

    rng.shuffle(all_samples)
    print(f"  Got {len(all_samples)} samples.", flush=True)

    if len(all_samples) < total_needed:
        print(f"  Warning: only {len(all_samples)} samples available, "
              f"need {total_needed}. Using all.")
        num_train = int(len(all_samples) * num_train / total_needed)
        num_eval = len(all_samples) - num_train

    train_samples = all_samples[:num_train]
    eval_samples = all_samples[num_train:num_train + num_eval]

    return train_samples, eval_samples


def save_jsonl(samples: list[dict], path: str) -> None:
    """Write samples to a JSONL file."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def print_stats(samples: list[dict], label: str) -> None:
    """Print summary statistics for a set of samples."""
    ratings = [s["puzzle_rating"] for s in samples]
    buckets = [(200, 800), (800, 1200), (1200, 1600),
               (1600, 2000), (2000, 2400), (2400, 2800)]
    print(f"\n{label}: {len(samples)} samples")
    print(f"  Rating range: {min(ratings)}-{max(ratings)}")
    for lo, hi in buckets:
        count = sum(1 for r in ratings if lo <= r < hi)
        print(f"  [{lo}, {hi}): {count} ({100 * count / len(ratings):.1f}%)")
    avg_legal = sum(len(s["legal_moves"]) for s in samples) / len(samples)
    print(f"  Avg legal moves per position: {avg_legal:.1f}")


if __name__ == "__main__":
    train, eval_ = preprocess()
    save_jsonl(train, TRAIN_PATH)
    save_jsonl(eval_, EVAL_PATH)
    print_stats(train, "Train")
    print_stats(eval_, "Eval")

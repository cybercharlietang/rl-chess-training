"""Evaluate a model on held-out chess puzzles."""

import argparse
import json

from rewards.format_reward import extract_answer_move, has_valid_tags, is_legal_move


def load_eval_data(path: str) -> list[dict]:
    """Load evaluation samples from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def evaluate_completions(
    samples: list[dict],
    completions: list[str],
) -> dict:
    """Score a list of model completions against puzzle solutions.

    Args:
        samples: eval dataset entries with 'solution_move', 'fen', 'puzzle_rating'
        completions: raw model output strings (one per sample)

    Returns:
        Dict with overall metrics and per-rating-bucket accuracy.
    """
    assert len(samples) == len(completions)

    correct = 0
    legal = 0
    format_ok = 0
    bucket_correct: dict[str, list[bool]] = {}
    buckets = [(200, 800), (800, 1200), (1200, 1600),
               (1600, 2000), (2000, 2400), (2400, 2800)]

    for sample, completion in zip(samples, completions):
        predicted = extract_answer_move(completion)
        solution = sample["solution_move"]
        fen = sample["fen"]
        rating = sample["puzzle_rating"]

        is_correct = (predicted == solution) if predicted else False
        correct += is_correct
        format_ok += int(has_valid_tags(completion) > 0)
        legal += int(is_legal_move(completion, fen) > 0)

        # Bucket by rating
        for lo, hi in buckets:
            if lo <= rating < hi:
                key = f"{lo}-{hi}"
                bucket_correct.setdefault(key, []).append(is_correct)
                break

    n = len(samples)
    results = {
        "num_samples": n,
        "accuracy": correct / n,
        "legal_move_rate": legal / n,
        "format_compliance_rate": format_ok / n,
        "accuracy_by_rating": {
            k: sum(v) / len(v) for k, v in sorted(bucket_correct.items())
        },
    }
    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({results['num_samples']} samples)")
    print(f"{'='*50}")
    print(f"  Puzzle accuracy:      {results['accuracy']:.3f}")
    print(f"  Legal move rate:      {results['legal_move_rate']:.3f}")
    print(f"  Format compliance:    {results['format_compliance_rate']:.3f}")
    print(f"\n  Accuracy by rating bucket:")
    for bucket, acc in results["accuracy_by_rating"].items():
        print(f"    {bucket}: {acc:.3f}")
    print(f"{'='*50}\n")


# ── GPU-dependent code below: model loading + inference ──────────────────


def generate_completions(model, tokenizer, samples, batch_size=8):
    """Generate model completions for eval samples.

    Stub — will be implemented when we have GPU access.
    Uses greedy decoding (temperature=0) for deterministic evaluation.
    """
    raise NotImplementedError("Requires GPU. Implement with model.generate().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default="data/eval.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    samples = load_eval_data(args.eval_data)
    print(f"Loaded {len(samples)} eval samples.")

    # TODO: Load model, generate completions, evaluate
    # completions = generate_completions(model, tokenizer, samples)
    # results = evaluate_completions(samples, completions)
    # print_results(results)
    print("Model loading/inference not yet implemented (needs GPU).")

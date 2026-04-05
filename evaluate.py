"""Evaluate a model on held-out chess puzzles."""

import argparse
import json
import os

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from prompts import build_chat_messages
from rewards.format_reward import extract_move, has_valid_tags, is_legal_move


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
        predicted = extract_move(completion)
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


def load_model_and_tokenizer(model_path: str, base_model: str | None = None):
    """Load model and tokenizer for evaluation.

    If base_model is provided, model_path is treated as a LoRA adapter dir.
    """
    if base_model:
        print(f"Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def generate_completions(
    model,
    tokenizer,
    samples: list[dict],
    batch_size: int = 8,
    max_new_tokens: int = 2048,
) -> list[str]:
    """Generate model completions for eval samples using greedy decoding."""
    completions = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch_samples = samples[i : i + batch_size]

        # Build prompts
        prompts = []
        for s in batch_samples:
            messages = build_chat_messages(s["fen"], s["legal_moves"])
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Tokenize with left-padding for batch generation
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens (strip the prompt)
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated_ids = output[prompt_len:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(completion)

    return completions


def save_detailed_results(
    samples: list[dict],
    completions: list[str],
    output_path: str,
) -> None:
    """Save per-sample results for the visualizer."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []
    for sample, completion in zip(samples, completions):
        predicted = extract_move(completion)
        solution = sample["solution_move"]
        fen = sample["fen"]
        is_correct = (predicted == solution) if predicted else False

        results.append({
            "fen": fen,
            "legal_moves": sample["legal_moves"],
            "solution_move": solution,
            "puzzle_rating": sample["puzzle_rating"],
            "completion": completion,
            "predicted_move": predicted,
            "correct": is_correct,
            "rewards": {
                "format": float(has_valid_tags(completion)),
                "legal": float(is_legal_move(completion, fen)),
            },
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path to LoRA adapter dir")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (required if --model is a LoRA adapter)")
    parser.add_argument("--eval_data", type=str, default="data/eval.jsonl")
    parser.add_argument("--output", type=str, default="outputs/eval_results.jsonl",
                        help="Save per-sample results JSONL to this path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N eval samples (default: all)")
    args = parser.parse_args()

    samples = load_eval_data(args.eval_data)
    if args.n is not None:
        samples = samples[:args.n]
    print(f"Loaded {len(samples)} eval samples.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.base_model)

    completions = generate_completions(
        model, tokenizer, samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    results = evaluate_completions(samples, completions)
    print_results(results)

    # Save detailed per-sample results for visualizer
    save_detailed_results(samples, completions, args.output)

    # Save summary results
    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")

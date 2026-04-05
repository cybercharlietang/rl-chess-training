"""Evaluate a model on held-out chess puzzles.

Uses data-parallel evaluation: each GPU loads a full model copy and
processes its share of the samples independently.
"""

import argparse
import json
import os
import math

import chess
import torch
import torch.multiprocessing as mp
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
    """Score completions against puzzle solutions."""
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


# ── Per-GPU worker ───────────────────────────────────────────────────────

def gpu_worker(
    gpu_id: int,
    model_name: str,
    base_model: str | None,
    samples: list[dict],
    max_new_tokens: int,
    batch_size: int,
    result_dict: dict,
):
    """Run generation on a single GPU for a subset of samples."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load model on this GPU
    if base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
        ).to(device)
        model = PeftModel.from_pretrained(model, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).to(device)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[GPU {gpu_id}] Model loaded, processing {len(samples)} samples")

    completions = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        prompts = []
        for s in batch:
            messages = build_chat_messages(s["fen"], s["legal_moves"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated_ids = output[prompt_len:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(completion)

        done = min(i + batch_size, len(samples))
        print(f"[GPU {gpu_id}] {done}/{len(samples)} done")

    result_dict[gpu_id] = completions
    print(f"[GPU {gpu_id}] Finished all {len(samples)} samples")


def generate_completions_parallel(
    model_name: str,
    base_model: str | None,
    samples: list[dict],
    max_new_tokens: int,
    batch_size_per_gpu: int,
    num_gpus: int,
) -> list[str]:
    """Generate completions in parallel across GPUs."""
    # Split samples across GPUs
    chunk_size = math.ceil(len(samples) / num_gpus)
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]

    # Shared dict for results
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, model_name, base_model, chunk,
                  max_new_tokens, batch_size_per_gpu, result_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Reassemble in order
    completions = []
    for gpu_id in range(len(chunks)):
        completions.extend(result_dict[gpu_id])

    return completions


# ── Single-GPU fallback ──────────────────────────────────────────────────

def generate_completions_single(
    model_name: str,
    base_model: str | None,
    samples: list[dict],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Generate completions on a single GPU (or pipeline-parallel with device_map=auto)."""
    if base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(model, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    completions = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i : i + batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                build_chat_messages(s["fen"], s["legal_moves"]),
                tokenize=False, add_generation_prompt=True,
            ) for s in batch
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            completion = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
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
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path to LoRA adapter dir")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (required if --model is a LoRA adapter)")
    parser.add_argument("--eval_data", type=str, default="data/eval.jsonl")
    parser.add_argument("--output", type=str, default="outputs/eval_results.jsonl",
                        help="Save per-sample results JSONL to this path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N eval samples (default: all)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs for data-parallel eval (default: all available)")
    args = parser.parse_args()

    samples = load_eval_data(args.eval_data)
    if args.n is not None:
        samples = samples[:args.n]
    print(f"Loaded {len(samples)} eval samples.")

    num_gpus = args.num_gpus or torch.cuda.device_count()

    if num_gpus > 1:
        print(f"Data-parallel eval across {num_gpus} GPUs, batch_size={args.batch_size}/GPU")
        completions = generate_completions_parallel(
            model_name=args.model,
            base_model=args.base_model,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            batch_size_per_gpu=args.batch_size,
            num_gpus=num_gpus,
        )
    else:
        print(f"Single-GPU eval, batch_size={args.batch_size}")
        completions = generate_completions_single(
            model_name=args.model,
            base_model=args.base_model,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    results = evaluate_completions(samples, completions)
    print_results(results)

    save_detailed_results(samples, completions, args.output)

    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")

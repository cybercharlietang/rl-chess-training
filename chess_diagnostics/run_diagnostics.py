"""Main entry point for chess LLM diagnostic tests."""

import argparse
import json
import os
import time

import chess
from tqdm import tqdm


def run_all(model_name: str, output_dir: str, tests_to_run: list[str] | None = None):
    """Run all diagnostic tests on a model."""
    from .model_utils import load_model, generate_answers_batch, extract_short_answer
    from . import test_fen_parsing, test_legal_moves, test_legality, test_consequences, test_rules_knowledge
    from .report import generate_html_report

    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    model, tokenizer = load_model(model_name)

    all_tests = {
        "fen_parsing": (test_fen_parsing, 25),
        "legal_moves": (test_legal_moves, 15),
        "legality": (test_legality, 25),
        "consequences": (test_consequences, 15),
        "rules_knowledge": (test_rules_knowledge, 50),
    }

    if tests_to_run:
        all_tests = {k: v for k, v in all_tests.items() if k in tests_to_run}

    all_samples = []
    all_metrics = []
    total_start = time.time()

    for test_key, (test_module, n_samples) in all_tests.items():
        print(f"\n{'='*60}")
        print(f"Running: {test_key} ({n_samples} samples)")
        print(f"{'='*60}")

        # Generate samples
        samples = test_module.generate_samples(n=n_samples)
        print(f"Generated {len(samples)} samples")

        # Run inference
        questions = [s["question"] for s in samples]
        start = time.time()
        raw_answers = generate_answers_batch(model, tokenizer, questions, batch_size=4, max_new_tokens=4096)
        elapsed = time.time() - start
        print(f"Inference: {elapsed:.1f}s ({elapsed/len(samples):.1f}s per sample)")

        # Score
        for s, raw in zip(samples, raw_answers):
            s["raw_answer"] = raw

            if test_key == "fen_parsing":
                s["is_correct"] = test_fen_parsing.score_answer(raw, s["correct_answer"])
            elif test_key == "legal_moves":
                board = chess.Board(s["fen"])
                piece_sq = chess.parse_square(s["square"])
                s["score"] = test_legal_moves.score_answer(raw, s["correct_moves"], board, piece_sq)
                s["is_correct"] = s["score"]["f1"] > 0.5
            elif test_key == "legality":
                s["is_correct"] = test_legality.score_answer(raw, s["correct_answer"])
            elif test_key == "consequences":
                s["is_correct"] = test_consequences.score_answer(raw, s["correct_answer"])
            elif test_key == "rules_knowledge":
                s["is_correct"] = test_rules_knowledge.score_answer(raw, s)

        # Compute metrics
        metrics = test_module.compute_metrics(samples)
        all_metrics.append(metrics)
        all_samples.extend(samples)

        # Print summary
        if "accuracy" in metrics:
            print(f"Accuracy: {metrics['accuracy']:.1%} (baseline: {metrics.get('baseline', 0):.1%})")
        if "mean_f1" in metrics:
            print(f"Mean F1: {metrics['mean_f1']:.1%}")
            print(f"Mean Precision: {metrics['mean_precision']:.1%}")
            print(f"Mean Recall: {metrics['mean_recall']:.1%}")

    total_time = (time.time() - total_start) / 60

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC SUMMARY — {model_name}")
    print(f"Total time: {total_time:.1f} min")
    print(f"{'='*60}")
    for m in all_metrics:
        score = m.get("accuracy", m.get("mean_f1", 0))
        baseline = m.get("baseline", 0)
        status = "ABOVE" if score > baseline * 1.5 else "AT" if score > baseline else "BELOW"
        print(f"  {m['test_name']:30s} {score:6.1%}  (baseline {baseline:.1%})  [{status}]")

    # Save JSON
    json_path = os.path.join(output_dir, "chess_diagnostics.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "total_time_min": total_time,
            "metrics": all_metrics,
            "samples": all_samples,
        }, f, indent=2, default=str)
    print(f"\nJSON saved to {json_path}")

    # Generate HTML report
    model_short = model_name.split("/")[-1]
    html_path = os.path.join(output_dir, f"diagnostics_{model_short}.html")
    generate_html_report(model_name, all_samples, all_metrics, total_time, html_path)

    # Also copy to project root for easy serving
    root_html = os.path.join(os.path.dirname(output_dir), f"diagnostics_{model_short}.html")
    generate_html_report(model_name, all_samples, all_metrics, total_time, root_html)

    return all_metrics, all_samples


def main():
    parser = argparse.ArgumentParser(description="Chess LLM Diagnostic Tests")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output_dir", type=str, default="results/diagnostics")
    parser.add_argument("--test", type=str, default=None,
                        choices=["fen_parsing", "legal_moves", "legality", "consequences", "rules_knowledge"],
                        help="Run a single test instead of all")
    args = parser.parse_args()

    tests = [args.test] if args.test else None
    run_all(args.model, args.output_dir, tests)


if __name__ == "__main__":
    main()

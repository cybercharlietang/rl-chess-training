"""Fast vLLM-based baseline eval for chess puzzles."""
import json, sys, time, argparse, os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/rl-chess-training")
from prompts import build_chat_messages
from rewards.format_reward import extract_move, has_valid_tags, is_legal_move

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="/workspace/rl-chess-training/data/eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--out", default="/workspace/rl-chess-training/outputs/qwen3_4b_baseline.jsonl")
    ap.add_argument("--gpu_mem", type=float, default=0.80)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading tokenizer + vLLM...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    max_model_len = args.max_tokens + 2048
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=max_model_len)

    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(samples)} samples", flush=True)

    prompts = []
    for s in samples:
        msgs = build_chat_messages(s["fen"], s["legal_moves"])
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=bool(args.thinking))
        prompts.append(p)

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] Generated {len(outs)} in {elapsed:.1f}s ({elapsed/len(outs):.2f}s/sample)", flush=True)

    correct = legal = fmt = truncated = 0
    results = []
    for s, o in zip(samples, outs):
        text = o.outputs[0].text
        ntok = len(o.outputs[0].token_ids)
        trunc = (ntok >= args.max_tokens)
        truncated += int(trunc)
        pred = extract_move(text)
        sol = s["solution_move"]
        is_correct = (pred == sol) if pred else False
        correct += int(is_correct)
        legal += int(is_legal_move(text, s["fen"]) > 0)
        fmt += int(has_valid_tags(text) > 0)
        results.append({
            "fen": s["fen"], "rating": s["puzzle_rating"], "solution": sol,
            "predicted": pred, "correct": is_correct, "truncated": trunc,
            "completion_tokens": ntok, "completion": text,
        })

    n = len(samples)
    stats = {
        "model": args.model, "thinking": bool(args.thinking), "max_tokens": args.max_tokens,
        "n": n, "accuracy": correct/n, "legal_rate": legal/n,
        "format_rate": fmt/n, "truncation_rate": truncated/n,
        "wall_s": elapsed,
    }
    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    with open(args.out, "w") as f:
        f.write(json.dumps(stats) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

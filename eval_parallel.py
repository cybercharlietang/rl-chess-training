"""Evaluate LoRA adapter — merge in-memory, data-parallel across all GPUs."""
import json, sys, time, argparse, os
sys.path.insert(0, "/workspace/rl-chess-training")

import transformers.integrations.tensor_parallel as _tp
if not hasattr(_tp, "EmbeddingParallel"):
    _tp.EmbeddingParallel = _tp.RowwiseParallel

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompts import build_chat_messages
from rewards.format_reward import extract_move, has_valid_tags, is_legal_move


def eval_worker(gpu_id, base_name, adapter_path, samples, max_tokens, batch_size, result_dict):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading + merging model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    print(f"[GPU {gpu_id}] Merged. Processing {len(samples)} samples, batch={batch_size}", flush=True)

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [tok.apply_chat_template(build_chat_messages(s["fen"], s["legal_moves"]),
                    tokenize=False, add_generation_prompt=True) for s in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens,
                                     do_sample=False, temperature=None, top_p=None)
        for j, (s, output) in enumerate(zip(batch, outputs)):
            prompt_len = inputs["input_ids"][j].shape[0]
            gen_ids = output[prompt_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            ntok = len(gen_ids)
            pred = extract_move(text)
            results.append({
                "fen": s["fen"], "rating": s["puzzle_rating"], "solution": s["solution_move"],
                "predicted": pred, "correct": (pred == s["solution_move"]) if pred else False,
                "truncated": ntok >= max_tokens, "completion_tokens": ntok, "completion": text,
            })
        done = min(i + batch_size, len(samples))
        print(f"[GPU {gpu_id}] {done}/{len(samples)} done", flush=True)

    result_dict[gpu_id] = results
    print(f"[GPU {gpu_id}] Finished.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="/workspace/rl-chess-training/data/eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break

    # Split across GPUs
    chunks = [[] for _ in range(args.num_gpus)]
    for i, s in enumerate(samples):
        chunks[i % args.num_gpus].append(s)

    print(f"Eval {len(samples)} puzzles across {args.num_gpus} GPUs (merge in-memory, batch={args.batch_size})", flush=True)
    t0 = time.time()

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()
    procs = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(target=eval_worker,
                       args=(gpu_id, args.base, args.adapter, chunks[gpu_id],
                             args.max_tokens, args.batch_size, result_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Reassemble in original order
    all_results = [None] * len(samples)
    for gpu_id in range(args.num_gpus):
        gpu_results = result_dict[gpu_id]
        for j, r in enumerate(gpu_results):
            orig_idx = gpu_id + j * args.num_gpus
            all_results[orig_idx] = r

    elapsed = time.time() - t0
    n = len(samples)
    correct = sum(1 for r in all_results if r and r["correct"])
    legal = sum(1 for r in all_results if r and is_legal_move(r["completion"], r["fen"]) > 0)
    fmt = sum(1 for r in all_results if r and has_valid_tags(r["completion"]) > 0)
    truncated = sum(1 for r in all_results if r and r["truncated"])

    stats = {"n": n, "accuracy": correct/n, "legal_rate": legal/n,
             "format_rate": fmt/n, "truncation_rate": truncated/n,
             "wall_s": elapsed, "adapter": args.adapter}

    print(f"\n{'='*50}")
    print(f"RESULTS ({n} puzzles, {elapsed:.1f}s)")
    print(f"{'='*50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    with open(args.out, "w") as f:
        f.write(json.dumps(stats) + "\n")
        for r in all_results:
            if r:
                f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()

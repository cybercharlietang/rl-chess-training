#!/bin/bash
# Full experiment: baseline eval → 50 GRPO steps → post-training eval → compare
set -e

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
EVAL_N=100
MAX_STEPS=50
OUTPUT_DIR="outputs/grpo_run"

echo "============================================"
echo "Chess-GRPO Experiment (8x H100 SXM)"
echo "Model: $MODEL"
echo "Eval samples: $EVAL_N"
echo "GRPO steps: $MAX_STEPS"
echo "============================================"

# Step 1: Baseline evaluation (zero-shot)
echo ""
echo ">>> Step 1: Baseline evaluation on $EVAL_N puzzles..."
python evaluate.py \
    --model "$MODEL" \
    --eval_data data/eval.jsonl \
    --n $EVAL_N \
    --output outputs/baseline_eval.jsonl \
    --max_new_tokens 8192

# Step 2: GRPO training (50 steps, dense Stockfish reward)
echo ""
echo ">>> Step 2: GRPO training ($MAX_STEPS steps, dense reward)..."
torchrun --nproc_per_node=8 train_grpo.py \
    --reward_mode dense \
    --max_steps $MAX_STEPS \
    --output_dir "$OUTPUT_DIR"

# Step 3: Post-training evaluation
echo ""
echo ">>> Step 3: Post-training evaluation on $EVAL_N puzzles..."
python evaluate.py \
    --model "$OUTPUT_DIR/final_adapter" \
    --base_model "$MODEL" \
    --eval_data data/eval.jsonl \
    --n $EVAL_N \
    --output outputs/trained_eval.jsonl \
    --max_new_tokens 8192

# Step 4: Print comparison
echo ""
echo ">>> Step 4: Results comparison"
python -c "
import json

with open('outputs/baseline_eval.jsonl.replace(\".jsonl\", \"_summary.json\")') as f:
    pass
# Use the summary files
baseline_path = 'outputs/baseline_eval_summary.json'
trained_path = 'outputs/trained_eval_summary.json'

# Fallback: try .jsonl → _summary.json naming
import os
for p in [baseline_path, trained_path]:
    if not os.path.exists(p):
        alt = p.replace('_summary.json', '.jsonl').replace('.jsonl', '_summary.json')
        if os.path.exists(alt):
            continue

with open(baseline_path) as f:
    baseline = json.load(f)
with open(trained_path) as f:
    trained = json.load(f)

print()
print('=' * 60)
print('BEFORE vs AFTER GRPO (50 steps, dense Stockfish reward)')
print('=' * 60)
print(f\"{'Metric':<25} {'Baseline':>10} {'Trained':>10} {'Delta':>10}\")
print('-' * 60)

for key in ['accuracy', 'legal_move_rate']:
    b = baseline.get(key, 0)
    t = trained.get(key, 0)
    d = t - b
    print(f'{key:<25} {b:>10.3f} {t:>10.3f} {d:>+10.3f}')

print()
print('Accuracy by rating:')
all_buckets = sorted(set(
    list(baseline.get('accuracy_by_rating', {}).keys()) +
    list(trained.get('accuracy_by_rating', {}).keys())
))
for bucket in all_buckets:
    b = baseline.get('accuracy_by_rating', {}).get(bucket, 0)
    t = trained.get('accuracy_by_rating', {}).get(bucket, 0)
    d = t - b
    print(f'  {bucket:<21} {b:>10.3f} {t:>10.3f} {d:>+10.3f}')
print('=' * 60)
"

# Step 5: Training curve from step logs
echo ""
echo ">>> Step 5: Training curve summary"
python -c "
import json

logs = []
log_path = '$OUTPUT_DIR/step_logs.jsonl'
try:
    with open(log_path) as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
except FileNotFoundError:
    print(f'No step logs found at {log_path}')
    exit()

if not logs:
    print('Step log file is empty.')
    exit()

print()
print('=' * 60)
print('TRAINING CURVE (per-step metrics)')
print('=' * 60)

# Find available metric keys
all_keys = set()
for log in logs:
    all_keys.update(log.keys())

reward_key = next((k for k in ['reward', 'rewards/mean', 'reward_mean'] if k in all_keys), None)
loss_key = next((k for k in ['loss', 'train_loss'] if k in all_keys), None)

print(f'Available metrics: {sorted(all_keys - {\"step\"})}')
print()

for log in logs[-10:]:  # last 10 steps
    step = log.get('step', '?')
    parts = [f'step={step:>4}']
    if loss_key and loss_key in log:
        parts.append(f'loss={log[loss_key]:.4f}')
    if reward_key and reward_key in log:
        parts.append(f'reward={log[reward_key]:.4f}')
    for k in sorted(log.keys()):
        if k not in ('step', loss_key, reward_key, 'epoch') and isinstance(log[k], (int, float)):
            parts.append(f'{k}={log[k]:.4f}')
    print('  '.join(parts))

print('=' * 60)
"

echo ""
echo ">>> Experiment complete. Results:"
echo "    outputs/baseline_eval.jsonl        - pre-training per-sample results"
echo "    outputs/baseline_eval_summary.json  - pre-training summary"
echo "    outputs/trained_eval.jsonl          - post-training per-sample results"
echo "    outputs/trained_eval_summary.json   - post-training summary"
echo "    $OUTPUT_DIR/step_logs.jsonl         - per-step training metrics"
echo "    $OUTPUT_DIR/final_adapter/          - trained LoRA adapter"
echo "    $OUTPUT_DIR/config.json             - experiment config"

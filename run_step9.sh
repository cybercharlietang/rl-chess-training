#!/bin/bash
# Step 9: Baseline eval (14B @ 8192 tokens) then GRPO training (50 steps)
set -e
cd /root/rl-chess-training
source .venv/bin/activate

echo "=============================================="
echo "Step 9a: Baseline eval (14B, 8192 tokens, 100 samples)"
echo "Started at: $(date)"
echo "=============================================="

python evaluate.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --eval_data data/eval.jsonl \
    --output outputs/baseline_14b_8192.jsonl \
    --batch_size 4 \
    --max_new_tokens 8192 \
    --num_samples 100 \
    2>&1 | tee outputs/baseline_14b_8192.log

echo ""
echo "=============================================="
echo "Step 9b: GRPO training (50 steps, dense rewards)"
echo "Started at: $(date)"
echo "=============================================="

python train_grpo.py \
    --reward_mode dense \
    --max_steps 50 \
    2>&1 | tee outputs/grpo_50step.log

echo ""
echo "=============================================="
echo "All done at: $(date)"
echo "=============================================="

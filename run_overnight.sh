#!/bin/bash
# Overnight GRPO training: 50 steps from r2 adapter
set -e
cd /root/rl-chess-training
source .venv/bin/activate

# ── Cleanup: kill any stale training processes and free VRAM ──
echo "Checking for stale GPU processes..."
STALE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n "$STALE_PIDS" ]; then
    echo "Killing stale GPU processes: $STALE_PIDS"
    for pid in $STALE_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
fi

VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
echo "VRAM after cleanup: ${VRAM_USED} MiB"
if [ "$VRAM_USED" -gt 1000 ]; then
    echo "ERROR: VRAM still occupied (${VRAM_USED} MiB). Aborting."
    exit 1
fi

echo "=============================================="
echo "Overnight GRPO Training (50 steps from r2 adapter)"
echo "Started at: $(date)"
echo "=============================================="

python train_grpo.py \
    --reward_mode dense \
    --max_steps 50 \
    --train_data data/train_easy.jsonl \
    --output_dir outputs/grpo_overnight \
    --adapter outputs/grpo_easy_r2/final_adapter \
    2>&1

echo ""
echo "=============================================="
echo "Training complete at: $(date)"
echo "=============================================="

# Run eval on the trained adapter (100 samples)
echo "Running eval on trained adapter..."
python evaluate.py \
    --model outputs/grpo_overnight/final_adapter \
    --eval_data data/eval.jsonl \
    --output outputs/eval_overnight.jsonl \
    --batch_size 4 \
    --max_new_tokens 8192 \
    --num_samples 100 \
    2>&1

echo "All done at: $(date)"

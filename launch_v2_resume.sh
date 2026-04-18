#!/bin/bash
# Launch GRPO v2 continued from adapter on 5x H200
set -e
MAX_STEPS=${1:-100}
LR=${2:-5e-6}
RUN_NAME=${3:-grpo_v2_easy_100step}
SAVE_STEPS=${4:-10}
ADAPTER=${5:-outputs/grpo_v2_easy/final_adapter}
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
LOG_DIR="outputs/${RUN_NAME}"

if [ -d /root/venv ]; then source /root/venv/bin/activate; else source /workspace/rl-chess-training/.venv/bin/activate; fi
cd /workspace/rl-chess-training
mkdir -p "${LOG_DIR}"

echo "=== Disk check ==="
df -h /workspace | tail -1

echo "=== vLLM server on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 nohup trl vllm-serve \
    --model "${MODEL}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --host 127.0.0.1 --port 8000 \
    > "${LOG_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 120); do
    curl -s -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1 && echo "vLLM ready $((i*5))s" && break
    sleep 5
done

echo "=== DDP training on GPUs 1-4, loading adapter from ${ADAPTER} ==="
CUDA_VISIBLE_DEVICES=1,2,3,4 PYTORCH_ALLOC_CONF=expandable_segments:True torchrun \
    --nproc_per_node=4 --master_port 29501 \
    train_grpo_v2.py \
    --reward_mode dense \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --run_name "${RUN_NAME}" \
    --save_steps "${SAVE_STEPS}" \
    --adapter "${ADAPTER}" \
    2>&1 | tee "${LOG_DIR}/train.log"

kill -9 ${VLLM_PID} 2>/dev/null
echo done

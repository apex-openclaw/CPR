#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# Optional: specify checkpoint (e.g., ./scripts/run_eval.sh checkpoint-471)
CHECKPOINT=${1:-""}

if [ -n "$CHECKPOINT" ]; then
    ADAPTER_PATH="outputs/qwen35-cpr-lora/${CHECKPOINT}"
    PRED_DIR="outputs/qwen35-cpr-lora/preds-${CHECKPOINT}"
    echo "Evaluating checkpoint: $ADAPTER_PATH"
else
    ADAPTER_PATH="outputs/qwen35-cpr-lora"
    PRED_DIR="outputs/qwen35-cpr-lora/preds-reasoning"
    echo "Evaluating final adapter: $ADAPTER_PATH"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Use cpreval env python (has vllm + sklearn)
PYTHON=${CPREVAL_PYTHON:-/root/anaconda3/envs/cpreval/bin/python}

# Step 1: Run inference with vLLM (fast batch generation)
echo "=== Running vLLM inference ==="
"$PYTHON" scripts/vllm_infer_reasoning.py \
    --model Qwen/Qwen3-4B \
    --adapter "$ADAPTER_PATH" \
    --dataset data/prepared/cpr_test.jsonl \
    --output "$PRED_DIR/generated_predictions.jsonl" \
    --max_new_tokens 4096 \
    --max_model_len 13000

# Step 2: Evaluate
echo ""
echo "=== Running evaluation ==="
"$PYTHON" scripts/eval_predictions.py \
    --predictions "$PRED_DIR/generated_predictions.jsonl" \
    --test_data data/prepared/test.jsonl \
    --output "$PRED_DIR/eval_results.json"

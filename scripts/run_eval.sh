#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

CHECKPOINT=${1:-""}

if [ -n "$CHECKPOINT" ]; then
    ADAPTER_PATH="outputs/qwen35-cpr-lora-newrun-nothink/${CHECKPOINT}"
    PRED_DIR="outputs/qwen35-cpr-lora-newrun-nothink/preds-${CHECKPOINT}"
    echo "Evaluating checkpoint: $ADAPTER_PATH"
else
    ADAPTER_PATH="outputs/qwen35-cpr-lora-newrun-nothink"
    PRED_DIR="outputs/qwen35-cpr-lora-newrun-nothink/preds"
    echo "Evaluating final adapter: $ADAPTER_PATH"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${CPREVAL_PYTHON:-python3}

for TASK in bio mesh moa target; do
    TEST_FILE="data/prepared/${TASK}_test.jsonl"
    if [ ! -f "$TEST_FILE" ]; then
        echo "Skipping $TASK: $TEST_FILE not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Task: $TASK"
    echo "=========================================="

    echo "=== Running vLLM inference for $TASK ==="
    "$PYTHON" scripts/vllm_infer.py \
        --model Qwen/Qwen3-4B \
        --adapter "$ADAPTER_PATH" \
        --dataset "$TEST_FILE" \
        --output "$PRED_DIR/${TASK}_predictions.jsonl" \
        --max_new_tokens 512 \
        --max_model_len 13000

    echo ""
    echo "=== Running evaluation for $TASK ==="
    "$PYTHON" scripts/eval_predictions.py \
        --predictions "$PRED_DIR/${TASK}_predictions.jsonl" \
        --test_data "$TEST_FILE" \
        --task_type "$TASK" \
        --output "$PRED_DIR/${TASK}_eval.json"
done

echo ""
echo "All evaluations complete. Results in: $PRED_DIR/"

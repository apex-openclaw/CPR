#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

# Optional: specify checkpoint (e.g., ./scripts/run_eval.sh checkpoint-800)
CHECKPOINT=${1:-""}
INFER_CONFIG="configs/infer_qwen35.yaml"

if [ -n "$CHECKPOINT" ]; then
    ADAPTER_PATH="outputs/qwen35-cpr-lora/${CHECKPOINT}"
    PRED_DIR="outputs/qwen35-cpr-lora/preds-${CHECKPOINT}"
    echo "Evaluating checkpoint: $ADAPTER_PATH"
else
    ADAPTER_PATH="outputs/qwen35-cpr-lora"
    PRED_DIR="outputs/qwen35-cpr-lora/preds"
    echo "Evaluating final adapter: $ADAPTER_PATH"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Step 1: Run inference
echo "=== Running inference ==="
llamafactory-cli predict \
    --stage evaluate \
    --model_name_or_path Qwen/Qwen3-4B \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen3 \
    --finetuning_type lora \
    --seed 42 \
    --dataset cpr_test \
    --dataset_dir data/prepared \
    --cutoff_len 8192 \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --do_predict true \
    --per_device_eval_batch_size 2 \
    --output_dir "$PRED_DIR"

# Step 2: Evaluate
echo ""
echo "=== Running evaluation ==="
python scripts/eval_predictions.py \
    --predictions "$PRED_DIR/generated_predictions.jsonl" \
    --test_data data/prepared/test.jsonl \
    --output "$PRED_DIR/eval_results.json"

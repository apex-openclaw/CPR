#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-"$PROJECT_ROOT/configs/sft_qwen35.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PATH=/root/anaconda3/envs/CPR13/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-/root/anaconda3/envs/CPR13/bin/llamafactory-cli}
"$LLAMAFACTORY_CLI" train "$CONFIG"

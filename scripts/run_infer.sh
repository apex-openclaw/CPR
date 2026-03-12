#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-"$PROJECT_ROOT/configs/infer_qwen35.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

llamafactory-cli predict "$CONFIG"

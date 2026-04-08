#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-"$PROJECT_ROOT/data_prep/joint_config.yaml"}
SEED=${SEED:-42}

python3 "$PROJECT_ROOT/data_prep/prepare_joint_dataset.py" --config "$CONFIG" --seed "$SEED"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-"$PROJECT_ROOT/data_prep/dataset_config.yaml"}
SEED=${SEED:-42}

PYTHON=${PYTHON:-/root/anaconda3/envs/CPR13/bin/python}
"$PYTHON" "$PROJECT_ROOT/data_prep/prepare_cpr_dataset.py" --config "$CONFIG" --seed "$SEED"

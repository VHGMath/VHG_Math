#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON:-python3}"
DOMAIN="${DOMAIN:?set DOMAIN=integration or DOMAIN=math}"

exec "${PYTHON_BIN}" scripts/train_solver.py \
  --role solver \
  --domain "${DOMAIN}" \
  --data_root "${DATA_ROOT:?set DATA_ROOT to the prepared RL data root}" \
  --model_path "${MODEL_PATH:?set MODEL_PATH to the solver initializer}" \
  --project_name "${PROJECT_NAME:-vhg_solver_rl}" \
  --experiment_name "${EXPERIMENT_NAME:-${DOMAIN}_solver}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints and rollout logs}" \
  --gpus "${GPUS:-8}" \
  "$@"

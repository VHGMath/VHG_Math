#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON:-python3}"

exec "${PYTHON_BIN}" scripts/eval_solver.py \
  --task "${TASK:?set TASK=integration or TASK=math}" \
  --records_file "${RECORDS_FILE:?set RECORDS_FILE to the evaluation JSONL}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR for evaluation outputs}" \
  --model_path "${MODEL_PATH:?set MODEL_PATH to the solver checkpoint}" \
  --visible_devices "${VISIBLE_DEVICES:?set VISIBLE_DEVICES, for example 0,1,2,3,4,5,6,7}" \
  --samples_per_problem "${SAMPLES_PER_PROBLEM:?set SAMPLES_PER_PROBLEM}" \
  "$@"

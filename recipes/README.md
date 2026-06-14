# Recipe Guide

Run all commands from the repository root. The recipes launch the release
training and evaluation stages; they expect data-construction outputs and model
checkpoints to be available as local paths.

## Install

```bash
conda create -y -n vhg_rl python=3.12
conda activate vhg_rl
bash env/install.sh
python env/check_env.py
```

Use `PYTHON=/path/to/env/bin/python` before any recipe if you want a specific
interpreter.

## Data Inputs

The release includes the seed pools used by the anonymous submission:

- `data/integration_seeds.jsonl`: 165 hard-verifier integration seeds.
- `data/general_math_seeds.jsonl`: 1,711 soft-verifier easy MATH seeds.

Large intermediate artifacts are not included. Training recipes expect parquet
data roots with this layout:

```text
/path/to/rl_data/
├── setter/
│   ├── train.parquet
│   └── test.parquet
└── solver/
    ├── train.parquet
    └── test.parquet
```

In the submission terminology, the `setter` split trains the problem setter.

## Hard-Verifier Setting

Train the integration problem setter:

```bash
DOMAIN=integration \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DATA_ROOT=/path/to/integration_rl_data \
MODEL_PATH=/path/to/setter/initializer \
SOLVER_CKPT_PATH=/path/to/solver/checkpoint \
OUTPUT_DIR=outputs/integration_setter_rl \
GPUS=8 \
bash recipes/train_setter.sh
```

Train the integration solver:

```bash
DOMAIN=integration \
DATA_ROOT=/path/to/integration_rl_data \
MODEL_PATH=/path/to/solver/initializer \
OUTPUT_DIR=outputs/integration_solver_rl \
GPUS=8 \
bash recipes/train_solver.sh
```

Evaluate an integration solver checkpoint. `RECORDS_FILE` should be a JSONL
file with `expr` or `integrand`, `var` or `variable`, and `question_type`
defaulting to `integration`.

```bash
TASK=integration \
RECORDS_FILE=/path/to/integration_eval.jsonl \
MODEL_PATH=/path/to/solver/checkpoint \
VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SAMPLES_PER_PROBLEM=64 \
OUTPUT_DIR=outputs/integration_solver_eval \
bash recipes/eval_solver.sh \
  --max_tokens 4096 \
  --gpu_memory_utilization 0.95
```

The hard-verifier service starts automatically for `DOMAIN=integration`
training and `TASK=integration` evaluation.

## Soft-Verifier Setting

Train the general-math problem setter:

```bash
DOMAIN=math \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DATA_ROOT=/path/to/math_rl_data \
MODEL_PATH=/path/to/setter/initializer \
SOLVER_CKPT_PATH=/path/to/solver/checkpoint \
SOFT_VERIFIER_BASE_URL=https://api.example.com/v1 \
SOFT_VERIFIER_API_KEY=your_api_key \
SOFT_VERIFIER_MODEL=your_judge_model \
OUTPUT_DIR=outputs/math_setter_rl \
GPUS=8 \
bash recipes/train_setter.sh
```

Train the general-math solver:

```bash
DOMAIN=math \
DATA_ROOT=/path/to/math_rl_data \
MODEL_PATH=/path/to/solver/initializer \
OUTPUT_DIR=outputs/math_solver_rl \
GPUS=8 \
bash recipes/train_solver.sh
```

Evaluate a general-math solver checkpoint. `RECORDS_FILE` should be a JSONL
file with `problem` or `derived_problem`, plus an answer field or a boxed answer
inside `solution`, `modified_solution`, `seed_solution`, or `original_solution`.

```bash
TASK=math \
RECORDS_FILE=/path/to/math_eval.jsonl \
MODEL_PATH=/path/to/solver/checkpoint \
VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SAMPLES_PER_PROBLEM=16 \
OUTPUT_DIR=outputs/math_solver_eval \
bash recipes/eval_solver.sh \
  --max_tokens 4096
```

For AMC and AIME-style benchmarks, use `--max_tokens 8192`.

## Outputs

Training writes VERL checkpoints, validation data, and rollout data under
`OUTPUT_DIR`. Solver evaluation writes:

- `evaluation_results.jsonl`: per-problem samples, correctness flags, and pass metrics.
- `statistics.json`: aggregate pass metrics and evaluation metadata.

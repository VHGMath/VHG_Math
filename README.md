# Anonymous Mathematical Reasoning Code Release

This repository is an anonymous academic code release for a mathematical
reasoning problem-generation framework. Author names, affiliations, the exact
submission title, and camera-ready citation metadata are intentionally omitted
during double-blind review.

The framework trains a problem setter with feedback from both a solver and a
verifier. This release keeps two submission settings separate:

- **Hard-verifier setting**: indefinite integration generation with a hard
  symbolic verifier.
- **Soft-verifier setting**: general math generation with a soft LLM-based
  verifier and local solver-difficulty checks.

The code is organized around the core paper concepts:

- `vhg/prompts.py`: versioned setter, solver, and soft-verifier prompts.
- `vhg/verify.py`: hard-verifier integration checks and soft-verifier format utilities.
- `vhg/score.py`: pass-rate, hardness, and reward scoring.
- `vhg/rewards.py`: VERL-compatible reward entrypoints.
- `vhg/generate.py`: generation and filtering helpers.
- `vhg/service.py`: standalone verifier service and client.
- `vhg/train.py`: thin training command builders.
- `vhg/evaluate.py`: solver evaluation helpers.

Large artifacts are intentionally not included. Provide model checkpoints,
generated problem pools, solver-training parquet files, logs, and W&B outputs
through local paths when running the recipes.

## Quick Start

Install the full release environment for local vLLM solver evaluation and
RL training:

```bash
conda create -y -n vhg_rl python=3.12
conda activate vhg_rl
bash env/install.sh
python env/check_env.py
```

The paper-facing recipe scripts correspond to the real training and evaluation
stages:

- `recipes/train_setter.sh`: train the problem setter.
- `recipes/train_solver.sh`: train the solver with prepared solver-RL data.
- `recipes/eval_solver.sh`: sample and score a solver checkpoint.

See [recipes/README.md](recipes/README.md) for copyable commands for the two
release settings.

Use `PYTHON=/path/to/python` before a recipe if you want to run with an
already-created environment:

```bash
PYTHON=/path/to/env/bin/python \
DOMAIN=integration \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DATA_ROOT=/path/to/integration_rl_data \
MODEL_PATH=/path/to/setter/initializer \
SOLVER_CKPT_PATH=/path/to/solver/checkpoint \
OUTPUT_DIR=outputs/integration_setter_rl \
bash recipes/train_setter.sh
```

## Data

The files under `data/` are the release-sized paper seed pools. They do not
include generated problem pools, checkpoints, logs, or solver-training
artifacts.

## Citation

During review, cite the associated anonymous submission. Camera-ready citation
metadata will be restored after the double-blind review period.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vhg.service import wait_for_correctness_service
from vhg.train import (
    build_setter_rl_config,
    build_solver_rl_config,
    build_verifier_service_command,
    build_verl_ppo_command,
    run_command,
)
from vhg.verify import ensure_latex_parser_available


VERIFIER_SERVICE_PORT = 5000
VERIFIER_SERVICE_STARTUP_TIMEOUT = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solver or setter RL.")
    parser.add_argument("--role", choices=("solver", "setter"), default="solver")
    parser.add_argument("--domain", choices=("integration", "math"), default="integration")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--project_name", default="vhg_solver_rl")
    parser.add_argument("--experiment_name", default="solver")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpus", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.domain == "integration" and args.role in {"solver", "setter"}:
        ensure_latex_parser_available()
    if args.role == "solver":
        config = build_solver_rl_config(
            data_root=args.data_root,
            model_path=args.model_path,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            n_gpus_per_node=args.gpus,
            domain=args.domain,
        )
    elif args.role == "setter":
        config = build_setter_rl_config(
            data_root=args.data_root,
            model_path=args.model_path,
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            n_gpus_per_node=args.gpus,
            domain=args.domain,
        )
    command = build_verl_ppo_command(config)
    if args.domain == "integration":
        service_command = build_verifier_service_command()
        os.environ["STANDALONE_SERVICE_PORT"] = str(VERIFIER_SERVICE_PORT)
        service_process = subprocess.Popen(service_command, cwd=str(REPO_ROOT))
        try:
            wait_for_correctness_service(
                f"http://localhost:{VERIFIER_SERVICE_PORT}",
                process=service_process,
                timeout_seconds=VERIFIER_SERVICE_STARTUP_TIMEOUT,
            )
            raise SystemExit(run_command(command, cwd=REPO_ROOT))
        finally:
            service_process.terminate()
            try:
                service_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service_process.kill()
    if args.domain == "math":
        os.environ.pop("STANDALONE_SERVICE_PORT", None)
    raise SystemExit(run_command(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()

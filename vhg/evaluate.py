from __future__ import annotations

from typing import Any

from .generate import (
    VHGRuntimeConfig,
    config_from_env,
    normalize_visible_devices,
    sample_local_solver_prompts,
)
from .prompts import make_math_solver_prompt, question_format_prompt, solver_messages
from .score import _check_items, summarize_solution_samples
from .utils import aggregate_pass_metrics, compute_pass_metrics, render_generation_prompt
from .verify import extract_last_boxed


def build_integration_solver_prompts(
    records: list[dict[str, Any]],
    tokenizer: Any = None,
    *,
    plain_prompt: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    prompts = []
    questions = []
    for item in records:
        expr = item.get("expr") or item.get("integrand")
        var = item.get("var") or item.get("variable")
        question_type = item.get("question_type", "integration")
        question = question_format_prompt(question_type, expr, var)
        benchmark_question = question_format_prompt(question_type, expr, var, latex=True)
        prompts.append(
            render_generation_prompt(
                solver_messages(benchmark_question),
                tokenizer=tokenizer,
                plain_prompt=plain_prompt,
            )
        )
        questions.append(
            {
                "question": question,
                "question_type": question_type,
                "expr": expr,
                "var": var,
                "raw_record": item,
            }
        )
    return prompts, questions


def evaluate_integration_samples(
    questions: list[dict[str, Any]],
    samples: list[list[str]],
) -> list[dict[str, Any]]:
    batch_items = []
    problem_indices = []
    for idx, (question, sample_list) in enumerate(zip(questions, samples)):
        for sample in sample_list:
            batch_items.append(
                {
                    "question": question["question"],
                    "question_type": question["question_type"],
                    "solution": sample,
                }
            )
            problem_indices.append(idx)
    flags = _check_items(batch_items)
    results = []
    for idx, (question, sample_list) in enumerate(zip(questions, samples)):
        problem_flags = [
            flags[i] for i, problem_idx in enumerate(problem_indices) if problem_idx == idx
        ]
        correct_count = sum(problem_flags)
        results.append(
            {
                **question,
                "samples": sample_list,
                "correct_flags": problem_flags,
                "correct_count": correct_count,
                "total_samples": len(problem_flags),
                "pass_at_k": compute_pass_metrics(correct_count, len(problem_flags)),
            }
        )
    return results


def build_math_solver_prompts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for idx, row in enumerate(records):
        problem = str(
            row.get("derived_problem")
            or row.get("problem")
            or row.get("original_problem")
            or row.get("seed_problem")
            or ""
        ).strip()
        reference_answer = str(
            row.get("modified_reference_boxed")
            or row.get("modified_solution_boxed")
            or row.get("reference_answer")
            or row.get("answer")
            or ""
        ).strip()
        if not reference_answer:
            reference_answer = extract_last_boxed(
                str(
                    row.get("modified_solution")
                    or row.get("solution")
                    or row.get("original_solution")
                    or row.get("seed_solution")
                    or ""
                )
            )
        entries.append(
            {
                "source_index": idx,
                "source_record_id": row.get("id"),
                "problem": problem,
                "reference_answer": reference_answer,
                "prompt": make_math_solver_prompt(problem),
                "raw_record": row,
            }
        )
    return entries


def build_solver_eval_config(
    *,
    model_path: str = "",
    visible_devices: str = "",
    gpu_memory_utilization: float | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> VHGRuntimeConfig:
    env_config = config_from_env()
    resolved_model_path = model_path.strip() or env_config.solver_ckpt_path
    resolved_visible_devices = normalize_visible_devices(
        visible_devices.strip() or env_config.solver_visible_devices
    )
    if not resolved_model_path:
        raise ValueError("Missing solver model path.")
    if not resolved_visible_devices:
        raise ValueError("Missing visible GPUs for solver evaluation.")
    return VHGRuntimeConfig(
        solver_ckpt_path=resolved_model_path,
        solver_visible_devices=resolved_visible_devices,
        solver_gpu_memory_utilization=(
            float(gpu_memory_utilization)
            if gpu_memory_utilization is not None
            else env_config.solver_gpu_memory_utilization
        ),
        solver_max_tokens=(
            int(max_tokens) if max_tokens is not None else env_config.solver_max_tokens
        ),
        solver_temperature=(
            float(temperature)
            if temperature is not None
            else env_config.solver_temperature
        ),
        solver_top_p=float(top_p) if top_p is not None else env_config.solver_top_p,
        solver_top_k=int(top_k) if top_k is not None else env_config.solver_top_k,
    )


def sample_solver_prompt_outputs(
    prompts: list[str],
    *,
    config: VHGRuntimeConfig,
    samples_per_problem: int,
) -> list[list[str]]:
    if not prompts:
        return []
    temperature = (
        0.0 if samples_per_problem <= 1 else float(config.solver_temperature)
    )
    top_p = None if samples_per_problem <= 1 else config.solver_top_p
    top_k = None if samples_per_problem <= 1 else config.solver_top_k
    indexed_prompts = [
        {"idx": str(idx), "prompt": prompt} for idx, prompt in enumerate(prompts)
    ]
    sampled = sample_local_solver_prompts(
        indexed_prompts,
        config=config,
        sample_num=int(samples_per_problem),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=int(config.solver_max_tokens),
    )
    return [sampled.get(str(idx), []) for idx in range(len(prompts))]


def evaluate_math_solver_samples(
    entries: list[dict[str, Any]],
    samples: list[list[str]],
    *,
    scoring_rule: str = "strict_single_boxed_answer",
) -> list[dict[str, Any]]:
    results = []
    for entry, sample_list in zip(entries, samples):
        results.append(
            {
                **summarize_solution_samples(
                    entry["problem"],
                    entry["reference_answer"],
                    sample_list,
                    scoring_rule=scoring_rule,
                ),
                "source_record_id": entry.get("source_record_id"),
                "raw_record": entry.get("raw_record"),
            }
        )
    return results


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_pass_metrics(results)

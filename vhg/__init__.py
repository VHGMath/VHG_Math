from __future__ import annotations


_EXPORTS = {
    "PROMPT_VERSION": ("vhg.prompts", "PROMPT_VERSION"),
    "make_math_setter_messages": ("vhg.prompts", "make_math_setter_messages"),
    "make_math_setter_prompt": ("vhg.prompts", "make_math_setter_prompt"),
    "make_math_solver_prompt": ("vhg.prompts", "make_math_solver_prompt"),
    "question_format_prompt": ("vhg.prompts", "question_format_prompt"),
    "question_solution_prompt": ("vhg.prompts", "question_solution_prompt"),
    "VHGRuntimeConfig": ("vhg.generate", "VHGRuntimeConfig"),
    "MathCandidateEvaluationInput": ("vhg.generate", "MathCandidateEvaluationInput"),
    "MathSeedRow": ("vhg.generate", "MathSeedRow"),
    "evaluate_generated_candidates_batched": (
        "vhg.generate",
        "evaluate_generated_candidates_batched",
    ),
    "sample_local_solver_prompts": ("vhg.generate", "sample_local_solver_prompts"),
    "compute_score_solver": ("vhg.score", "compute_score_solver"),
    "compute_score_solver_batched": ("vhg.score", "compute_score_solver_batched"),
    "analyze_hard_verifier_generated_pairs_batched": (
        "vhg.score",
        "analyze_hard_verifier_generated_pairs_batched",
    ),
    "enrich_hard_verifier_solver_metrics_batched": (
        "vhg.score",
        "enrich_hard_verifier_solver_metrics_batched",
    ),
    "evaluate_math_candidate_record": ("vhg.score", "evaluate_math_candidate_record"),
    "score_hard_verifier_setter_candidate": ("vhg.score", "score_hard_verifier_setter_candidate"),
    "score_hard_verifier_setter_candidates_batched": (
        "vhg.score",
        "score_hard_verifier_setter_candidates_batched",
    ),
    "CorrectnessClient": ("vhg.service", "CorrectnessClient"),
    "check_correctness": ("vhg.service", "check_correctness"),
    "check_batch": ("vhg.verify", "check_batch"),
    "correctness": ("vhg.verify", "correctness"),
    "find_expr": ("vhg.verify", "find_expr"),
    "parse_soft_verifier_response": ("vhg.verify", "parse_soft_verifier_response"),
    "parse_math_setter_response": ("vhg.verify", "parse_math_setter_response"),
    "parse_symbolic_generation_question": (
        "vhg.verify",
        "parse_symbolic_generation_question",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'vhg' has no attribute {name!r}")
    import importlib

    module_name, attr_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value

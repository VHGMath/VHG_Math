from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .prompts import (
    make_math_solver_prompt,
    make_math_setter_messages,
    make_soft_verifier_prompt,
    question_format_prompt,
)
from .utils import load_jsonl, render_generation_prompt, save_jsonl
from .verify import (
    extract_single_parseable_boxed_answer,
    format_math_setter_response,
    parse_soft_verifier_response,
    parse_math_setter_response,
)


@dataclass(frozen=True)
class MathSeedRow:
    idx: int
    seed_problem: str
    seed_solution: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MathCandidateEvaluationInput:
    seed: MathSeedRow
    reasoning: str
    derived_problem: str
    modified_solution: str
    generation_trace: dict[str, Any]
    objective: str = "hard"
    solver_eval_samples: int = 4
    solver_temperature: float | None = None
    solver_top_p: float | None = None
    solver_top_k: int | None = None
    solver_max_tokens: int | None = None


@dataclass(frozen=True)
class VHGRuntimeConfig:
    gen_base_url: str = ""
    gen_api_key: str = ""
    gen_model: str = "gpt-5-nano"
    gen_max_tokens: int = 4096
    soft_verifier_base_url: str = ""
    soft_verifier_api_key: str = ""
    soft_verifier_model: str = ""
    soft_verifier_max_tokens: int = 512
    solver_ckpt_path: str = ""
    solver_visible_devices: str = ""
    solver_gpu_memory_utilization: float = 0.9
    solver_max_tokens: int = 2048
    solver_temperature: float = 0.7
    solver_top_p: float = 0.95
    solver_top_k: int | None = None
    solver_eval_samples: int = 4
    max_retries: int = 3


def normalize_visible_devices(visible_devices: str) -> str:
    return ",".join(
        device.strip()
        for device in str(visible_devices or "").split(",")
        if device.strip()
    )


def is_local_openai_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def config_from_env() -> VHGRuntimeConfig:
    gen_base_url = os.getenv("GEN_BASE_URL", "")
    gen_api_key = os.getenv("GEN_API_KEY", "")
    gen_model = os.getenv("GEN_MODEL", "gpt-5-nano")
    soft_verifier_base_url = os.getenv("SOFT_VERIFIER_BASE_URL", gen_base_url)
    soft_verifier_api_key = os.getenv("SOFT_VERIFIER_API_KEY", gen_api_key)
    soft_verifier_model = os.getenv("SOFT_VERIFIER_MODEL", gen_model)
    solver_ckpt_path = os.getenv("SOLVER_CKPT_PATH", os.getenv("BASE_MODEL_PATH", ""))
    solver_visible_devices = normalize_visible_devices(
        os.getenv("SOLVER_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", ""))
    )
    if not soft_verifier_api_key and is_local_openai_base_url(soft_verifier_base_url):
        soft_verifier_api_key = "EMPTY"
    return VHGRuntimeConfig(
        gen_base_url=gen_base_url,
        gen_api_key=gen_api_key,
        gen_model=gen_model,
        gen_max_tokens=int(os.getenv("GEN_MAX_TOKENS", "4096")),
        soft_verifier_base_url=soft_verifier_base_url,
        soft_verifier_api_key=soft_verifier_api_key,
        soft_verifier_model=soft_verifier_model,
        soft_verifier_max_tokens=int(os.getenv("SOFT_VERIFIER_MAX_TOKENS", "512")),
        solver_ckpt_path=solver_ckpt_path,
        solver_visible_devices=solver_visible_devices,
        solver_gpu_memory_utilization=float(os.getenv("SOLVER_GPU_MEMORY_UTIL", "0.9")),
        solver_max_tokens=int(os.getenv("SOLVER_MAX_TOKENS", "2048")),
        solver_temperature=float(os.getenv("SOLVER_TEMPERATURE", "0.7")),
        solver_top_p=float(os.getenv("SOLVER_TOP_P", "0.95")),
        solver_top_k=(
            int(os.getenv("SOLVER_TOP_K"))
            if os.getenv("SOLVER_TOP_K", "").strip()
            else None
        ),
        solver_eval_samples=int(os.getenv("SOLVER_EVAL_SAMPLES", "4")),
        max_retries=int(
            os.getenv(
                "SOFT_VERIFIER_MAX_RETRIES",
                os.getenv("MAX_RETRIES", "3"),
            )
        ),
    )


def ensure_config(
    config: VHGRuntimeConfig,
    *,
    need_generation: bool,
    need_solver: bool = True,
) -> None:
    missing = []
    if need_generation:
        if not config.gen_api_key:
            missing.append("GEN_API_KEY")
        if not config.gen_base_url:
            missing.append("GEN_BASE_URL")
    if not config.soft_verifier_api_key and not is_local_openai_base_url(config.soft_verifier_base_url):
        missing.append("SOFT_VERIFIER_API_KEY or GEN_API_KEY")
    if not config.soft_verifier_base_url:
        missing.append("SOFT_VERIFIER_BASE_URL or GEN_BASE_URL")
    if need_solver:
        if not config.solver_ckpt_path:
            missing.append("SOLVER_CKPT_PATH or BASE_MODEL_PATH")
        if not normalize_visible_devices(config.solver_visible_devices):
            missing.append("SOLVER_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES")
    if missing:
        raise ValueError("Missing runtime configuration: " + ", ".join(missing))


def resolve_solver_sampling(
    config: VHGRuntimeConfig,
    *,
    n: int,
) -> tuple[float, float | None, int | None]:
    if n <= 1:
        return 0.0, None, None
    return config.solver_temperature, config.solver_top_p, config.solver_top_k


def build_openai_http_client(
    *,
    max_connections: int,
    timeout_seconds: float = 120.0,
) -> httpx.AsyncClient:
    connection_limit = max(32, int(max_connections))
    keepalive_limit = max(20, min(connection_limit, 256))
    timeout = httpx.Timeout(
        connect=min(30.0, timeout_seconds),
        read=timeout_seconds,
        write=timeout_seconds,
        pool=timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=connection_limit,
        max_keepalive_connections=keepalive_limit,
    )
    return httpx.AsyncClient(timeout=timeout, trust_env=False, limits=limits)


def load_math_seeds_from_file(
    path: str | Path,
) -> list[MathSeedRow]:
    rows: list[MathSeedRow] = []
    seen_ids: set[int] = set()
    for obj in load_jsonl(path):
        seed_problem = str(
            obj.get("seed_problem") or obj.get("original_problem") or obj.get("problem") or ""
        ).strip()
        seed_solution = str(
            obj.get("seed_solution")
            or obj.get("original_solution")
            or obj.get("solution")
            or ""
        ).strip()
        if not seed_problem or not seed_solution:
            continue
        _, targets = extract_single_parseable_boxed_answer(seed_solution)
        if not targets:
            continue
        raw_id = obj.get("id", obj.get("seed_id", len(rows)))
        seed_id = int(raw_id)
        if seed_id in seen_ids:
            raise ValueError(f"Seed file {path} contains duplicate id={seed_id}")
        seen_ids.add(seed_id)
        metadata = obj.get("metadata", {})
        rows.append(
            MathSeedRow(
                idx=seed_id,
                seed_problem=seed_problem,
                seed_solution=seed_solution,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return rows


class RealAPIBackend:
    def __init__(self, config: VHGRuntimeConfig):
        import openai

        self.config = config
        gen_connections = int(os.getenv("GEN_HTTP_MAX_CONNECTIONS", "128"))
        soft_verifier_connections = int(os.getenv("SOFT_VERIFIER_HTTP_MAX_CONNECTIONS", "512"))
        timeout_seconds = float(os.getenv("SOFT_VERIFIER_HTTP_TIMEOUT_SECONDS", "120"))
        self.gen_client = None
        if config.gen_base_url and config.gen_api_key:
            self.gen_client = openai.AsyncOpenAI(
                base_url=config.gen_base_url,
                api_key=config.gen_api_key,
                timeout=timeout_seconds,
                http_client=build_openai_http_client(
                    max_connections=gen_connections,
                    timeout_seconds=timeout_seconds,
                ),
            )
        self.soft_verifier_client = openai.AsyncOpenAI(
            base_url=config.soft_verifier_base_url,
            api_key=config.soft_verifier_api_key,
            timeout=timeout_seconds,
            http_client=build_openai_http_client(
                max_connections=soft_verifier_connections,
                timeout_seconds=timeout_seconds,
            ),
        )

    async def _chat_text(
        self,
        client,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if top_k is not None:
                    kwargs["extra_body"] = {"top_k": top_k}
                try:
                    resp = await client.chat.completions.create(**kwargs)
                except Exception:
                    if "extra_body" not in kwargs:
                        raise
                    kwargs.pop("extra_body", None)
                    resp = await client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError("OpenAI-compatible text call failed.") from last_error

    async def generate_candidate(
        self,
        seed: MathSeedRow,
        *,
        objective: str,
    ) -> dict[str, Any]:
        if self.gen_client is None:
            raise RuntimeError("Generation client is not configured.")
        response = await self._chat_text(
            self.gen_client,
            self.config.gen_model,
            make_math_setter_messages(
                seed.seed_problem,
                seed.seed_solution,
                objective=objective,
            ),
            max_tokens=self.config.gen_max_tokens,
            temperature=0.7,
        )
        payload = parse_math_setter_response(response) or {}
        return {
            "reasoning": str(payload.get("reasoning", "")).strip(),
            "derived_problem": str(payload.get("derived_problem", "")).strip(),
            "modified_solution": str(payload.get("modified_solution", "")).strip(),
            "generation_trace": {
                "objective": objective,
                "prompt_version": "tagged_v1",
            },
        }

    async def verify_pair(
        self,
        seed_problem: str,
        seed_solution: str,
        derived_problem: str,
        modified_solution: str,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": "Follow the requested tagged format exactly."},
            {
                "role": "user",
                "content": make_soft_verifier_prompt(
                    seed_problem=seed_problem,
                    seed_solution=seed_solution,
                    derived_problem=derived_problem,
                    modified_solution=modified_solution,
                ),
            },
        ]
        for _ in range(self.config.max_retries):
            response = await self._chat_text(
                self.soft_verifier_client,
                self.config.soft_verifier_model,
                messages,
                max_tokens=self.config.soft_verifier_max_tokens,
                temperature=0.0,
            )
            parsed = parse_soft_verifier_response(response)
            if parsed is not None:
                _, payload = parsed
                return payload
        return {
            "valid_problem": False,
            "valid_solution": False,
            "seed_anchored": False,
            "not_trivial_copy": False,
            "complete_final_answer": False,
            "reason": "soft_verifier_malformed_tagged_response",
        }


def build_backend(config: VHGRuntimeConfig):
    return RealAPIBackend(config)


def _effective_solver_config(
    config: VHGRuntimeConfig,
    candidate: MathCandidateEvaluationInput,
) -> VHGRuntimeConfig:
    return replace(
        config,
        solver_temperature=(
            config.solver_temperature
            if candidate.solver_temperature is None
            else float(candidate.solver_temperature)
        ),
        solver_top_p=(
            config.solver_top_p
            if candidate.solver_top_p is None
            else float(candidate.solver_top_p)
        ),
        solver_top_k=(
            config.solver_top_k
            if candidate.solver_top_k is None
            else int(candidate.solver_top_k)
        ),
        solver_max_tokens=(
            config.solver_max_tokens
            if candidate.solver_max_tokens is None
            else int(candidate.solver_max_tokens)
        ),
    )


def sample_local_solver_prompts(
    indexed_prompts: list[dict[str, str]],
    *,
    config: VHGRuntimeConfig,
    sample_num: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
) -> dict[str, list[str]]:
    if not indexed_prompts:
        return {}
    import ray
    from vllm import LLM, SamplingParams

    visible_devices = normalize_visible_devices(config.solver_visible_devices)
    if not visible_devices:
        raise ValueError(
            "SOLVER_VISIBLE_DEVICES is empty after parsing; expected comma-separated GPU ids."
        )
    if not config.solver_ckpt_path:
        raise ValueError("Missing solver checkpoint path for local solver sampling.")

    @ray.remote(num_gpus=0)
    class SolverSampler:
        def __init__(self, model_path: str, gpu_memory_utilization: float, cuda_device: str, rank: int):
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            python_bin = str(Path(sys.executable).resolve().parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin and python_bin not in path_parts:
                os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])
            base_port = int(os.environ.get("SOLVER_MASTER_PORT_BASE", "30000"))
            port_stride = int(os.environ.get("SOLVER_MASTER_PORT_STRIDE", "53"))
            init_retries = int(os.environ.get("SOLVER_LLM_INIT_RETRIES", "5"))
            last_error = None
            for attempt in range(init_retries):
                vllm_port = base_port + rank * port_stride + attempt
                if vllm_port > 65535:
                    vllm_port = 30000 + (vllm_port - 30000) % 30000
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["VLLM_PORT"] = str(vllm_port)
                try:
                    self.llm = LLM(
                        model=model_path,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=gpu_memory_utilization,
                        trust_remote_code=True,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < init_retries:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise
            raise RuntimeError("Failed to initialize solver sampler") from last_error

        def generate(self, batch: list[dict[str, str]], sampling_kwargs: dict[str, Any]):
            sampling_params = SamplingParams(**sampling_kwargs)
            outputs = self.llm.generate([entry["prompt"] for entry in batch], sampling_params)
            return {
                entry["idx"]: [completion.text for completion in output.outputs]
                for entry, output in zip(batch, outputs)
            }

    gpu_list = [device.strip() for device in visible_devices.split(",") if device.strip()]
    ray.init(ignore_reinit_error=True)
    actors = [
        SolverSampler.remote(
            config.solver_ckpt_path,
            float(config.solver_gpu_memory_utilization),
            gpu,
            rank,
        )
        for rank, gpu in enumerate(gpu_list)
    ]
    sampling_kwargs: dict[str, Any] = {
        "n": int(sample_num),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    def start(rank: int) -> int:
        floor = len(indexed_prompts) // len(actors)
        remainder = len(indexed_prompts) % len(actors)
        return rank * floor + min(rank, remainder)

    try:
        shards = [
            indexed_prompts[start(rank) : start(rank + 1)]
            for rank in range(len(actors))
        ]
        futures = [
            actor.generate.remote(shard, sampling_kwargs)
            for actor, shard in zip(actors, shards)
            if shard
        ]
        results: dict[str, list[str]] = {}
        for shard_result in ray.get(futures, timeout=float(os.getenv("SOLVER_RAY_GET_TIMEOUT_S", "3600"))):
            results.update(shard_result)
        return results
    finally:
        for actor in actors:
            ray.kill(actor)


_sample_local_solver_prompts = sample_local_solver_prompts


def _sample_local_solver_outputs(
    indexed_problems: list[dict[str, str]],
    *,
    config: VHGRuntimeConfig,
    sample_num: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
) -> dict[str, list[str]]:
    prompt_entries = [
        {
            "idx": item["idx"],
            "prompt": render_generation_prompt(
                [{"role": "user", "content": make_math_solver_prompt(item["problem"])}],
                plain_prompt=True,
            ),
        }
        for item in indexed_problems
    ]
    return sample_local_solver_prompts(
        prompt_entries,
        config=config,
        sample_num=sample_num,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )


async def evaluate_generated_candidates_batched(
    candidates: list[MathCandidateEvaluationInput],
    *,
    backend,
    soft_verifier_concurrency: int = 16,
    run_solver_eval: bool = True,
    solver_sampler=None,
) -> list[dict[str, Any]]:
    from .score import evaluate_math_candidate_record

    if not candidates:
        return []

    config = getattr(backend, "config", config_from_env())
    results: list[dict[str, Any] | None] = [None] * len(candidates)
    verifier_entries = []

    for position, candidate in enumerate(candidates):
        local_record = evaluate_math_candidate_record(
            seed_problem=candidate.seed.seed_problem,
            seed_solution=candidate.seed.seed_solution,
            reasoning=candidate.reasoning,
            derived_problem=candidate.derived_problem,
            modified_solution=candidate.modified_solution,
            verifier_payload=None,
            objective=candidate.objective,
            metadata=candidate.seed.metadata,
            generation_trace=candidate.generation_trace,
            record_id=candidate.seed.idx,
        )
        if local_record.get("verification", {}).get("reason") != "missing_verifier_payload":
            results[position] = local_record
            continue
        verifier_entries.append({"position": position, "candidate": candidate})

    semaphore = asyncio.Semaphore(max(1, int(soft_verifier_concurrency)))

    async def verify_one(entry: dict[str, Any]):
        candidate = entry["candidate"]
        async with semaphore:
            try:
                payload = await backend.verify_pair(
                    candidate.seed.seed_problem,
                    candidate.seed.seed_solution,
                    candidate.derived_problem,
                    candidate.modified_solution,
                )
            except Exception as exc:
                payload = {
                    "valid_problem": False,
                    "valid_solution": False,
                    "seed_anchored": False,
                    "not_trivial_copy": False,
                    "complete_final_answer": False,
                    "reason": f"soft_verifier_api_error: {type(exc).__name__}: {str(exc)[:240]}",
                }
        return entry, payload

    verifier_results = await asyncio.gather(*(verify_one(entry) for entry in verifier_entries))
    verified_entries = []
    for entry, verifier_payload in verifier_results:
        position = entry["position"]
        candidate = entry["candidate"]
        record = evaluate_math_candidate_record(
            seed_problem=candidate.seed.seed_problem,
            seed_solution=candidate.seed.seed_solution,
            reasoning=candidate.reasoning,
            derived_problem=candidate.derived_problem,
            modified_solution=candidate.modified_solution,
            verifier_payload=verifier_payload,
            objective=candidate.objective,
            metadata=candidate.seed.metadata,
            generation_trace=candidate.generation_trace,
            record_id=candidate.seed.idx,
        )
        results[position] = record
        if record.get("verified_valid"):
            verified_entries.append(
                {
                    "position": position,
                    "candidate": candidate,
                    "record": record,
                    "original_reference_boxed": extract_single_parseable_boxed_answer(
                        candidate.seed.seed_solution
                    )[0],
                }
            )
    if not verified_entries or not run_solver_eval:
        return [record for record in results if record is not None]

    grouped_entries: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    effective_configs: dict[tuple[Any, ...], VHGRuntimeConfig] = {}
    for entry in verified_entries:
        candidate = entry["candidate"]
        effective_config = _effective_solver_config(config, candidate)
        eval_temperature, eval_top_p, eval_top_k = resolve_solver_sampling(
            effective_config,
            n=int(candidate.solver_eval_samples),
        )
        key = (
            int(candidate.solver_eval_samples),
            eval_temperature,
            eval_top_p,
            eval_top_k,
            int(effective_config.solver_max_tokens),
        )
        grouped_entries[key].append(entry)
        effective_configs[key] = effective_config

    for key, entries in grouped_entries.items():
        sample_num, eval_temperature, eval_top_p, eval_top_k, max_tokens = key
        effective_config = effective_configs[key]
        indexed_problems = []
        original_idx_by_seed: dict[tuple[int, str, str], str] = {}
        for entry in entries:
            candidate = entry["candidate"]
            seed_key = (
                int(candidate.seed.idx),
                candidate.seed.seed_problem,
                entry["original_reference_boxed"],
            )
            if seed_key not in original_idx_by_seed:
                original_idx_by_seed[seed_key] = f"seed:{candidate.seed.idx}:original"
                indexed_problems.append(
                    {
                        "idx": original_idx_by_seed[seed_key],
                        "variant": "original",
                        "problem": candidate.seed.seed_problem,
                    }
                )
            entry["original_eval_idx"] = original_idx_by_seed[seed_key]
            indexed_problems.append(
                {
                    "idx": f"{entry['position']}:modified",
                    "variant": "modified",
                    "problem": candidate.derived_problem,
                }
            )

        if solver_sampler is not None:
            solutions_map = solver_sampler(
                indexed_problems=indexed_problems,
                config=effective_config,
                sample_num=sample_num,
                temperature=eval_temperature,
                top_p=eval_top_p,
                top_k=eval_top_k,
                max_tokens=max_tokens,
            )
        else:
            solutions_map = _sample_local_solver_outputs(
                indexed_problems,
                config=effective_config,
                sample_num=sample_num,
                temperature=eval_temperature,
                top_p=eval_top_p,
                top_k=eval_top_k,
                max_tokens=max_tokens,
            )

        for entry in entries:
            position = entry["position"]
            candidate = entry["candidate"]
            record = evaluate_math_candidate_record(
                seed_problem=candidate.seed.seed_problem,
                seed_solution=candidate.seed.seed_solution,
                reasoning=candidate.reasoning,
                derived_problem=candidate.derived_problem,
                modified_solution=candidate.modified_solution,
                verifier_payload=results[position].get("verification", {}),
                original_solver_outputs=solutions_map.get(entry["original_eval_idx"], []),
                modified_solver_outputs=solutions_map.get(f"{position}:modified", []),
                objective=candidate.objective,
                metadata=candidate.seed.metadata,
                generation_trace=candidate.generation_trace,
                record_id=candidate.seed.idx,
            )
            results[position] = record

    return [record for record in results if record is not None]


async def generate_math_candidates(
    seeds: list[MathSeedRow],
    *,
    backend,
    objective: str = "hard",
    concurrency: int = 16,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def generate_one(seed: MathSeedRow) -> dict[str, Any]:
        async with semaphore:
            candidate = await backend.generate_candidate(seed, objective=objective)
            return {
                "id": seed.idx,
                "seed_problem": seed.seed_problem,
                "seed_solution": seed.seed_solution,
                **candidate,
                "text": format_math_setter_response(
                    reasoning=candidate.get("reasoning", ""),
                    derived_problem=candidate.get("derived_problem", ""),
                    modified_solution=candidate.get("modified_solution", ""),
                ),
            }

    return await asyncio.gather(*(generate_one(seed) for seed in seeds))


def load_integration_seed_questions(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for row in rows:
        row.setdefault("question_type", "integration")
        row["question"] = question_format_prompt(
            row["question_type"],
            row.get("expr") or row.get("integrand"),
            row.get("var") or row.get("variable"),
        )
    return rows


def sample_records(records: list[dict[str, Any]], limit: int | None, seed: int = 42) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, limit)


def save_generated_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    save_jsonl(path, rows)

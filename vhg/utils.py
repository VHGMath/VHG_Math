from __future__ import annotations

import json
import os
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import yaml


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def render_generation_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any = None,
    *,
    plain_prompt: bool = False,
) -> str:
    if plain_prompt or tokenizer is None:
        if len(messages) == 1 and messages[0].get("role", "user") == "user":
            return messages[0].get("content", "")
        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def pass_at_k(total_samples: int, correct: int, k: int) -> float:
    if total_samples == 0:
        return 0.0
    if total_samples - correct < k:
        return 1.0
    product = 1.0
    for m in range(total_samples - correct + 1, total_samples + 1):
        product *= 1.0 - k / m
    return 1.0 - product


def compute_pass_metrics(correct: int, total_samples: int) -> dict[str, float]:
    if total_samples == 0:
        return {}
    return {
        f"pass@{k}": pass_at_k(total_samples, correct, k)
        for k in range(1, total_samples + 1)
    }


def aggregate_pass_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_samples = sum(int(row.get("total_samples", 0)) for row in rows)
    total_correct = sum(int(row.get("correct_count", 0)) for row in rows)
    sample_accuracy = total_correct / total_samples if total_samples else 0.0
    max_k = max((int(row.get("total_samples", 0)) for row in rows), default=0)
    average_pass = {}
    for k in range(1, max_k + 1):
        key = f"pass@{k}"
        values = [
            float(row["pass_at_k"][key])
            for row in rows
            if key in row.get("pass_at_k", {})
        ]
        if values:
            average_pass[key] = mean(values)
    return {
        "total_questions": len(rows),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "sample_accuracy": sample_accuracy,
        "overall_accuracy": sample_accuracy,
        **average_pass,
    }

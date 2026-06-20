"""Retrieval-quality metrics comparing returned context to golden supporting text."""

from __future__ import annotations

import re
from typing import Iterable, Union


def normalize_context_text(text: Union[str, list, None]) -> str:
    """Flatten search context payloads into comparable plain text."""
    if text is None:
        return ""
    if isinstance(text, list):
        parts = [normalize_context_text(item) for item in text]
        return "\n".join(part for part in parts if part)
    if isinstance(text, dict):
        for key in ("text", "content", "context", "payload"):
            if key in text and text[key]:
                return normalize_context_text(text[key])
        return " ".join(str(value) for value in text.values())
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text) if token}


def context_recall(retrieved: Union[str, list, None], golden: Union[str, list, None]) -> float:
    """Fraction of golden lines (or sentences) found in retrieved context."""
    golden_text = normalize_context_text(golden)
    retrieved_text = normalize_context_text(retrieved)
    if not golden_text:
        return 0.0

    units = _split_golden_units(golden_text)
    if not units:
        return 0.0

    hits = sum(1 for unit in units if unit in retrieved_text)
    return hits / len(units)


def context_overlap_f1(retrieved: Union[str, list, None], golden: Union[str, list, None]) -> float:
    """Token-level F1 between retrieved and golden context."""
    retrieved_tokens = _token_set(normalize_context_text(retrieved))
    golden_tokens = _token_set(normalize_context_text(golden))
    if not retrieved_tokens and not golden_tokens:
        return 0.0
    if not retrieved_tokens or not golden_tokens:
        return 0.0

    overlap = retrieved_tokens & golden_tokens
    precision = len(overlap) / len(retrieved_tokens)
    recall = len(overlap) / len(golden_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _split_golden_units(golden_text: str) -> list[str]:
    lines = [line.strip() for line in golden_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return [line.lower() for line in lines]

    sentences = re.split(r"(?<=[.!?])\s+", golden_text)
    sentences = [sentence.strip().lower() for sentence in sentences if sentence.strip()]
    return sentences or [golden_text]


def aggregate_metric(values: Iterable[float]) -> dict[str, float]:
    """Return mean/min/max for a metric series."""
    series = list(values)
    if not series:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    return {
        "mean": sum(series) / len(series),
        "min": min(series),
        "max": max(series),
        "count": float(len(series)),
    }

"""SQuAD-style token F1 — the LoCoMo paper's headline metric.

Normalization follows the SQuAD / LoCoMo evaluation scripts: lower-case, strip punctuation,
drop English articles, collapse whitespace. No third-party dependency (the harness's other
F1 metric imports deepeval at module load).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Optional

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def token_f1(prediction: str | None, gold: str | None) -> tuple[float, float, float]:
    """Return ``(f1, precision, recall)`` over normalized whitespace tokens."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class LocomoF1Metric:
    """Deepeval-free F1 metric with the ``measure`` / ``score`` / ``reason`` protocol."""

    def __init__(self) -> None:
        self.score: float | None = None
        self.reason: str | None = None

    def measure(self, test_case: Any) -> float:
        f1, precision, recall = token_f1(test_case.actual_output, test_case.expected_output)
        self.score = f1
        self.reason = f"F1: {f1:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f})"
        return f1

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

"""Token F1 exactly as the official LoCoMo scorer computes it.

Port of ``eval_question_answering`` in ``task_eval/evaluation.py`` of
https://github.com/snap-research/locomo, so the number is comparable with the paper:

- normalisation drops commas, punctuation, case, the articles *and* ``and``;
- tokens are Porter-stemmed before matching ("painted" == "painting");
- multi-hop (category 1): both answers are split on commas and every gold part takes the
  best F1 over the predicted parts, then the parts are averaged;
- open-domain (category 3): only the gold text before the first ``;`` is used;
- adversarial (category 5): no F1 at all — 1.0 when the answer contains ``not mentioned``
  or ``no information available``, else 0.0;
- single-hop / temporal (categories 4 / 2): plain stemmed F1.

``metrics/f1.py`` keeps the SQuAD-style F1 the harness used before; both are reported.
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from typing import Any, Optional

from cognee.eval_framework.benchmark_adapters.locomo_adapter import (
    CATEGORY_NAMES,
    is_abstention,
)

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCTUATION = set(string.punctuation)
_NAME_TO_CATEGORY = {name: number for number, name in CATEGORY_NAMES.items()}


def _stemmer():
    try:
        from nltk.stem import PorterStemmer
    except ImportError as error:  # pragma: no cover - environmental
        raise ImportError(
            'f1_locomo needs nltk (Porter stemmer). Install it with: pip install "cognee[evals]"'
        ) from error
    return PorterStemmer()


def normalize_answer_locomo(text: str | None) -> str:
    """Official LoCoMo normalisation (``normalize_answer`` in evaluation.py)."""
    if not text:
        return ""
    text = text.replace(",", "")
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def _stemmed_tokens(text: str | None, stemmer) -> list[str]:
    return [stemmer.stem(token) for token in normalize_answer_locomo(text).split()]


def f1_single(prediction: str | None, gold: str | None, stemmer=None) -> float:
    """``f1_score`` from evaluation.py: stemmed token F1, 0 when nothing overlaps."""
    stemmer = stemmer or _stemmer()
    pred_tokens = _stemmed_tokens(prediction, stemmer)
    gold_tokens = _stemmed_tokens(gold, stemmer)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_multi_answer(prediction: str | None, gold: str | None, stemmer=None) -> float:
    """``f1`` from evaluation.py: comma-split sub-answers, best match per gold part."""
    stemmer = stemmer or _stemmer()
    predictions = [part.strip() for part in (prediction or "").split(",")]
    golds = [part.strip() for part in (gold or "").split(",")]
    return statistics.fmean(
        max(f1_single(pred, gold_part, stemmer) for pred in predictions) for gold_part in golds
    )


def abstention_score(prediction: str | None) -> float:
    return 1.0 if is_abstention(prediction) else 0.0


def resolve_category(metadata: dict[str, Any] | None) -> int | None:
    """Category number from metadata: explicit ``category`` first, else the type name."""
    metadata = metadata or {}
    category = metadata.get("category")
    if category is not None:
        try:
            return int(category)
        except (TypeError, ValueError):
            pass
    return _NAME_TO_CATEGORY.get(str(metadata.get("question_type", "")))


def locomo_f1(prediction: str | None, gold: str | None, category: int | None) -> tuple[float, str]:
    """Return ``(score, rule)`` using the official per-category rule."""
    if category == 5:
        return abstention_score(prediction), "abstention"
    stemmer = _stemmer()
    if category == 1:
        return f1_multi_answer(prediction, gold, stemmer), "multi_answer_f1"
    if category == 3:
        gold = (gold or "").split(";")[0].strip()
        return f1_single(prediction, gold, stemmer), "stemmed_f1(first_gold_part)"
    return f1_single(prediction, gold, stemmer), "stemmed_f1"


class LocomoOfficialF1Metric:
    """Official-protocol F1 with the ``measure`` / ``score`` / ``reason`` protocol."""

    def __init__(self) -> None:
        self.score: float | None = None
        self.reason: str | None = None

    def measure(self, test_case: Any) -> float:
        category = resolve_category(getattr(test_case, "additional_metadata", None))
        score, rule = locomo_f1(test_case.actual_output, test_case.expected_output, category)
        self.score = score
        self.reason = f"F1 (LoCoMo official, {rule}): {score:.2f}"
        return score

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

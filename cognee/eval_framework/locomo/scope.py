"""The questions a LoCoMo evaluation answers, judges and reports.

A question is in scope unless it is open-domain (it needs knowledge from outside the conversation)
or an audit marked its gold answer as corrupted. The audited questions are listed in
``report_artifacts/gold_exclusions.json`` by conversation and by position in the conversation's
``qa`` list; the ``without_corrupted_golds`` variant leaves out every audit set together.

``answer.py`` drops out-of-scope questions before retrieval, answering and judging, and
``run_locomo_eval.py`` refuses to summarise a run directory that still holds any.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GOLD_EXCLUSIONS = (
    Path(__file__).resolve().parent / "report_artifacts" / "gold_exclusions.json"
)
GOLD_EXCLUSION_VARIANT = "without_corrupted_golds"
EXCLUDED_QUESTION_TYPES = ("open_domain",)

QuestionKey = tuple[int, int]  # (conversation_index, question_idx)


def load_gold_exclusions(path: Path) -> dict[str, set[QuestionKey]]:
    """``{variant: {(conversation_index, question_idx), ...}}`` from a gold-exclusions file.

    The file lists question sets under ``sets.<name>.questions`` and, under ``variants``, which
    sets each variant removes together. A malformed file raises ``ValueError``.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        sets = {
            name: {_question_key(question) for question in block["questions"]}
            for name, block in document["sets"].items()
        }
        return {
            variant: set().union(*(sets[name] for name in names))
            for variant, names in document["variants"].items()
        }
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: malformed gold exclusions ({error!r})") from error


def _question_key(question: dict[str, Any]) -> QuestionKey:
    key = (question["conversation_index"], question["question_idx"])
    if not all(isinstance(part, int) and not isinstance(part, bool) for part in key):
        raise ValueError(f"question ids must be integers: {question}")
    return key


@dataclass(frozen=True)
class EvaluationScope:
    """The audit exclusions of one gold-exclusions file, plus the excluded question types."""

    gold_exclusions: Path
    gold_exclusions_sha256: str
    excluded_questions: frozenset[QuestionKey]

    @classmethod
    def load(cls, path: Path = DEFAULT_GOLD_EXCLUSIONS) -> EvaluationScope:
        variants = load_gold_exclusions(path)
        if GOLD_EXCLUSION_VARIANT not in variants:
            raise ValueError(f"{path}: no {GOLD_EXCLUSION_VARIANT!r} variant")
        return cls(
            gold_exclusions=path.resolve(),
            gold_exclusions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            excluded_questions=frozenset(variants[GOLD_EXCLUSION_VARIANT]),
        )

    def includes(self, conversation_index: int, question_idx: int, question_type: str) -> bool:
        return (
            question_type not in EXCLUDED_QUESTION_TYPES
            and (conversation_index, question_idx) not in self.excluded_questions
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "gold_exclusions": str(self.gold_exclusions),
            "gold_exclusions_sha256": self.gold_exclusions_sha256,
            "gold_exclusion_variant": GOLD_EXCLUSION_VARIANT,
            "excluded_question_types": list(EXCLUDED_QUESTION_TYPES),
        }

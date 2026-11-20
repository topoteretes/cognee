import hashlib
import json
from enum import Enum
from typing import Any

from cognee.eval_framework.integrity.errors import RunIntegrityError


class AnswerStatus(str, Enum):
    """Outcome of answering one question, stored on the answer record as ``status``."""

    OK = "ok"
    RETRIEVAL_ERROR = "retrieval_error"
    LLM_ERROR = "llm_error"
    EMPTY = "empty"

    @property
    def failed(self) -> bool:
        return self in (AnswerStatus.RETRIEVAL_ERROR, AnswerStatus.LLM_ERROR)

    @property
    def scorable(self) -> bool:
        """Errored answers are not answers and must never reach an evaluator.

        An empty answer is real model output (a miss): it is scored, but counted separately.
        """
        return not self.failed


LEGACY_ERROR_PREFIX = "ERROR:"


def answer_status(record: dict[str, Any]) -> AnswerStatus:
    """The one place that decides whether an answer record failed.

    Records written before ``status`` existed count as ok, except those whose answer is the
    ``"ERROR: ..."`` text the old code stored for a failure: those are ``llm_error`` (the
    failed phase was not recorded).
    """
    if "status" not in record:
        answer = record.get("answer")
        legacy_failure = isinstance(answer, str) and answer.startswith(LEGACY_ERROR_PREFIX)
        return AnswerStatus.LLM_ERROR if legacy_failure else AnswerStatus.OK
    raw = record["status"]
    try:
        return AnswerStatus(raw)
    except ValueError:
        raise RunIntegrityError(
            f"Unknown answer status {raw!r} on question {record.get('question')!r}"
        ) from None


def question_id(question: Any, golden_answer: Any) -> str:
    """Identity of a question across the questions, answers and metrics files.

    Questions carry no id of their own and the files name the golden answer differently
    (``answer`` vs ``golden_answer``), so identity is a digest of question and golden answer.
    """
    payload = json.dumps([question, golden_answer], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def record_id(record: dict[str, Any]) -> str:
    """Identity of an answer or metrics record."""
    return question_id(record.get("question"), record.get("golden_answer"))


def source_question_id(question: dict[str, Any]) -> str:
    """Identity of an entry of the questions file."""
    return question_id(question.get("question"), question.get("answer"))


def count_statuses(records: list[dict[str, Any]]) -> dict[str, int]:
    """``{total, ok, empty, failed}`` over answer or metrics records."""
    counts = {"total": len(records), "ok": 0, "empty": 0, "failed": 0}
    for record in records:
        status = answer_status(record)
        if status.failed:
            counts["failed"] += 1
        elif status is AnswerStatus.EMPTY:
            counts["empty"] += 1
        else:
            counts["ok"] += 1
    return counts

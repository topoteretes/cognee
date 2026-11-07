from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel

from cognee.eval_framework.integrity.status import record_id, source_question_id


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class Finding(BaseModel):
    severity: Severity
    code: str
    message: str
    question_id: str | None = None


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _duplicates(artifact: str, ids: Counter) -> list[Finding]:
    return [
        Finding(
            severity=Severity.WARNING,
            code=f"duplicate_{artifact}",
            message=f"{artifact} contains the same question {count} times",
            question_id=qid,
        )
        for qid, count in ids.items()
        if count > 1
    ]


def _alignment(
    upstream_name: str, upstream: Counter, downstream_name: str, downstream: Counter
) -> list[Finding]:
    """Multiset comparison: every upstream question needs one downstream entry, and every
    downstream entry needs an upstream question. Surplus duplicates surface as orphans."""
    findings = [
        Finding(
            severity=Severity.ERROR,
            code=f"missing_{downstream_name}",
            message=f"question in {upstream_name} has no entry in {downstream_name}",
            question_id=qid,
        )
        for qid in (upstream - downstream)
    ]
    findings += [
        Finding(
            severity=Severity.ERROR,
            code=f"orphan_{downstream_name}",
            message=f"entry in {downstream_name} has no matching question in {upstream_name}",
            question_id=qid,
        )
        for qid in (downstream - upstream)
    ]
    return findings


def validate_run(
    questions: list[dict[str, Any]] | None = None,
    answers: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> list[Finding]:
    """Cross-file consistency checks over whichever run artifacts are given.

    Artifacts are matched by question identity (digest of question and golden answer). Pass
    only the artifacts that exist; absent ones are not compared. Checks: missing and orphaned
    entries between consecutive files, duplicates within a file, and empty golden answers.
    A failed answer still has a metrics entry (with no scores), so every question stays visible.
    """
    findings: list[Finding] = []
    counters: dict[str, Counter] = {}

    for name, entries, identify, golden_key in (
        ("questions", questions, source_question_id, "answer"),
        ("answers", answers, record_id, "golden_answer"),
        ("metrics", metrics, record_id, "golden_answer"),
    ):
        if entries is None:
            continue
        counters[name] = Counter(identify(entry) for entry in entries)
        findings += _duplicates(name, counters[name])
        findings += [
            Finding(
                severity=Severity.ERROR,
                code=f"empty_golden_answer_{name}",
                message=f"entry in {name} has an empty golden answer; it cannot be scored",
                question_id=identify(entry),
            )
            for entry in entries
            if _empty(entry.get(golden_key))
        ]

    previous: tuple[str, Counter] | None = None
    for name in ("questions", "answers", "metrics"):
        if name not in counters:
            continue
        if previous is not None:
            findings += _alignment(previous[0], previous[1], name, counters[name])
        previous = (name, counters[name])

    return findings

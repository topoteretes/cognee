import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from cognee.eval_framework.integrity.coverage import MetricCoverage
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.fingerprint import result_config
from cognee.eval_framework.integrity.io import atomic_write_json
from cognee.eval_framework.integrity.status import (
    AnswerStatus,
    answer_status,
    count_statuses,
    record_id,
)
from cognee.eval_framework.integrity.validate import Finding

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"

STAGES = ("corpus", "answers", "evaluation", "dashboard")


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class QuestionFailure(BaseModel):
    question_id: str
    status: AnswerStatus
    error: str | None = None


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    note: str | None = None
    # Entries in the stage's output artifact; ok/empty/failed only apply to the answers stage.
    total: int | None = None
    ok: int | None = None
    empty: int | None = None
    failed: int | None = None
    failures: list[QuestionFailure] = Field(default_factory=list)
    coverage: dict[str, MetricCoverage] | None = None


class RunManifest(BaseModel):
    """What a run did, stage by stage, and how much of it can be trusted.

    Written next to the run's artifacts after every stage transition, so an interrupted run
    still says how far it got.
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_at: str
    updated_at: str
    config: dict[str, Any]
    stages: dict[str, StageRecord] = Field(
        default_factory=lambda: {stage: StageRecord() for stage in STAGES}
    )
    # Digest of everything that determines the answers (see fingerprint.py); set when the answers
    # stage starts, and the only thing that lets a later run reuse the answer journal.
    answers_fingerprint: str | None = None
    findings: list[Finding] = Field(default_factory=list)

    @classmethod
    def new(cls, params: dict[str, Any]) -> "RunManifest":
        now = _now()
        return cls(
            created_at=now,
            updated_at=now,
            config=result_config(params),
        )

    @classmethod
    def load(cls, path: str) -> "RunManifest":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as error:
            raise RunIntegrityError(f"Run manifest {path} is corrupt: {error}") from error
        if not isinstance(raw, dict) or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            found = raw.get("schema_version") if isinstance(raw, dict) else None
            raise RunIntegrityError(
                f"Run manifest {path} has schema version {found!r}; "
                f"this cognee reads version {MANIFEST_SCHEMA_VERSION}"
            )
        try:
            return cls.model_validate(raw)
        except ValidationError as error:
            raise RunIntegrityError(f"Run manifest {path} is malformed: {error}") from error

    def save(self, path: str) -> None:
        self.updated_at = _now()
        atomic_write_json(path, self.model_dump(mode="json"), indent=2)

    def require_same_config(self, params: dict[str, Any]) -> None:
        """Refuse to resume a run whose result-determining config has changed."""
        current = json.loads(json.dumps(result_config(params), default=str))
        changed = sorted(
            k for k in current.keys() | self.config.keys() if self.config.get(k) != current.get(k)
        )
        if changed:
            details = ", ".join(
                f"{k}: {self.config.get(k)!r} -> {current.get(k)!r}" for k in changed
            )
            raise RunIntegrityError(f"Cannot resume, config changed since the manifest ({details})")

    def begin(self, stage: str) -> None:
        """Start a stage with a clean record, so a re-run keeps nothing from the previous one."""
        self.stages[stage] = StageRecord(status=StageStatus.RUNNING, started_at=_now())

    def finish(self, stage: str, status: StageStatus, **fields: Any) -> None:
        record = self.stages[stage]
        record.status = status
        record.finished_at = _now()
        for name, value in fields.items():
            setattr(record, name, value)

    def skip(self, stage: str, note: str) -> None:
        self.stages[stage].status = StageStatus.SKIPPED
        self.stages[stage].note = note


def manifest_path_for(run_dir: str) -> str:
    return os.path.join(run_dir, MANIFEST_FILENAME) if run_dir else MANIFEST_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_answers(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage fields for an answers artifact: status counts and one entry per failed question."""
    failures = []
    for answer in answers:
        status = answer_status(answer)
        if status.failed:
            failures.append(
                QuestionFailure(
                    question_id=record_id(answer), status=status, error=answer.get("error")
                )
            )
    return {**count_statuses(answers), "failures": failures}


def enforce_failure_rate(failed: int, total: int, max_rate: float | None, what: str) -> None:
    """Strict mode: raise when more than ``max_rate`` of ``total`` failed.

    ``max_rate`` is the user's own threshold (0.0 tolerates no failure); ``None`` disables the
    check, in which case failures are reported but never change the outcome.
    """
    if max_rate is None or total == 0:
        return
    if failed / total > max_rate:
        raise RunIntegrityError(
            f"{failed}/{total} {what} ({failed / total:.1%}) exceed the allowed rate {max_rate:.1%}"
        )

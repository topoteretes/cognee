"""What one maintenance job did after one pipeline run.

Same status vocabulary as an improve stage (``StageResult``): ``completed``
(the job did work), ``already_completed`` (nothing to do), ``skipped`` (a gate
said no; ``reason`` says why) and ``errored`` (the job raised; the run it
followed is unaffected).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from cognee.modules.operations import scrub_error_message

JobStatus = Literal["completed", "already_completed", "skipped", "errored"]

# Skip reasons the runner itself produces (jobs add their own).
REASON_DISABLED_BY_CONFIG = "disabled_by_config"
REASON_BACKEND_UNSUPPORTED = "backend_unsupported"
# The job's gate raised. When a job cannot tell whether it should run, it does
# not: a maintenance job may delete things, so the safe default is to wait for
# the next run (improve's gates, which only save work, fail open instead).
REASON_GATE_ERRORED = "gate_errored"
# A store-scoped job after one dataset of a multi-dataset run, with multi-user
# off: every dataset shares one store, so the job runs once, after the last.
REASON_SHARED_STORE = "shared_store_runs_after_last_dataset"


class JobResult(BaseModel):
    """What one maintenance job did in one run."""

    job: str
    status: JobStatus
    reason: str | None = None  # required when skipped
    error: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    duration_ms: int = 0

    @model_validator(mode="after")
    def _skipped_needs_reason(self) -> "JobResult":
        if self.status == "skipped" and not self.reason:
            raise ValueError(f"maintenance job '{self.job}' is skipped without a reason")
        return self

    @classmethod
    def completed(cls, job: str, **counts: int) -> "JobResult":
        return cls(job=job, status="completed", counts=dict(counts))

    @classmethod
    def already_completed(cls, job: str, **counts: int) -> "JobResult":
        return cls(job=job, status="already_completed", counts=dict(counts))

    @classmethod
    def skipped(cls, job: str, reason: str) -> "JobResult":
        return cls(job=job, status="skipped", reason=reason)

    @classmethod
    def errored(cls, job: str, error: Any) -> "JobResult":
        return cls(job=job, status="errored", error=_error_text(error))

    @property
    def has_failures(self) -> bool:
        """Errored, or did its work with some parts failing (``*_errors`` counts)."""
        return self.status == "errored" or any(
            value > 0 for key, value in self.counts.items() if key.endswith("_errors")
        )

    def summary(self) -> str:
        """One line for the run's log: ``name status (reason or counts)``."""
        detail = (
            self.reason
            or self.error
            or ", ".join(f"{key}={value}" for key, value in self.counts.items())
        )
        return f"{self.job} {self.status}" + (f" ({detail})" if detail else "")


def _error_text(error: Any) -> str:
    """Redacted like a pipeline run's error: a job's exception can carry a
    connection string or credentials, and this text is logged."""
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    return scrub_error_message(text) or type(error).__name__

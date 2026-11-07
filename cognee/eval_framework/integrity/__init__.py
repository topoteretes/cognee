"""Run integrity for the eval framework: explicit answer status, a run manifest, metric
coverage, cross-file validation and resumable answering."""

from cognee.eval_framework.integrity.checkpoint import AnswerCheckpoint
from cognee.eval_framework.integrity.coverage import MetricCoverage, compute_metric_coverage
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.manifest import (
    RunManifest,
    StageRecord,
    StageStatus,
    enforce_failure_rate,
)
from cognee.eval_framework.integrity.status import AnswerStatus
from cognee.eval_framework.integrity.validate import Finding, Severity, validate_run

__all__ = [
    "AnswerCheckpoint",
    "AnswerStatus",
    "Finding",
    "MetricCoverage",
    "RunIntegrityError",
    "RunManifest",
    "Severity",
    "StageRecord",
    "StageStatus",
    "compute_metric_coverage",
    "enforce_failure_rate",
    "validate_run",
]

"""The maintenance jobs, in the order they run.

``DEFAULT_JOBS`` is the single place a job is registered; ``test_registry``
pins its contents and order. Jobs run one after another, so a job listed later
sees what an earlier one left.
"""

from collections.abc import Iterable, Sequence

from .job import BaseMaintenanceJob
from .jobs import VectorCompactionJob

DEFAULT_JOBS: list[BaseMaintenanceJob] = [
    VectorCompactionJob(),
]


def job_names(jobs: Sequence[BaseMaintenanceJob] = DEFAULT_JOBS) -> list[str]:
    return [job.name for job in jobs]


def validate_jobs(jobs: Sequence[BaseMaintenanceJob]) -> None:
    """Every job has a unique, non-empty name and triggers on at least one pipeline."""
    names = job_names(jobs)
    if any(not name for name in names):
        raise ValueError("every maintenance job needs a name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate maintenance job names: {duplicates}")
    without_pipelines = [job.name for job in jobs if not job.pipelines]
    if without_pipelines:
        raise ValueError(f"maintenance jobs with no triggering pipeline: {without_pipelines}")


def validate_jobs_disabled(disabled: Iterable[str], jobs: Sequence[BaseMaintenanceJob]) -> None:
    unknown = sorted(set(disabled) - set(job_names(jobs)))
    if unknown:
        raise ValueError(
            f"MAINTENANCE_JOBS_DISABLED names unknown job(s) {unknown}; "
            f"valid names: {job_names(jobs)}"
        )


validate_jobs(DEFAULT_JOBS)

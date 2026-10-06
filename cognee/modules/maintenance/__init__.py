"""Maintenance jobs: bounded upkeep that runs after a completed pipeline run.

See ``job.BaseMaintenanceJob`` for the contract and how to add a job,
``registry.DEFAULT_JOBS`` for the jobs, and ``runner.run_maintenance`` for how
they run.
"""

from .config import MaintenanceConfig, get_maintenance_config
from .job import BaseMaintenanceJob, MaintenanceContext
from .registry import DEFAULT_JOBS, job_names, validate_jobs, validate_jobs_disabled
from .result import (
    REASON_BACKEND_UNSUPPORTED,
    REASON_DISABLED_BY_CONFIG,
    REASON_GATE_ERRORED,
    JobResult,
)
from .runner import run_maintenance

__all__ = [
    "DEFAULT_JOBS",
    "REASON_BACKEND_UNSUPPORTED",
    "REASON_DISABLED_BY_CONFIG",
    "REASON_GATE_ERRORED",
    "BaseMaintenanceJob",
    "JobResult",
    "MaintenanceConfig",
    "MaintenanceContext",
    "get_maintenance_config",
    "job_names",
    "run_maintenance",
    "validate_jobs",
    "validate_jobs_disabled",
]

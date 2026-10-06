"""Configuration owned by the maintenance runner.

Only the runner's own knob lives here. A job's feature flags and limits stay
with the subsystem it maintains (vector compaction reads ``VectorConfig``), so
there is one place to configure each thing.

Environment variables (prefix ``MAINTENANCE_``)::

    MAINTENANCE_JOBS_DISABLED=a,b      # csv of job names to skip
"""

from functools import lru_cache
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class MaintenanceConfig(BaseSettings):
    """Settings for the maintenance runner (env prefix ``MAINTENANCE_``)."""

    # Job names (see ``registry.DEFAULT_JOBS``) to skip with reason
    # ``disabled_by_config``. Read as a comma-separated string.
    jobs_disabled: Annotated[list[str], NoDecode] = []

    model_config = SettingsConfigDict(env_prefix="MAINTENANCE_", extra="ignore")

    @field_validator("jobs_disabled", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(part).strip() for part in value if str(part).strip()]


@lru_cache
def get_maintenance_config() -> MaintenanceConfig:
    """The env-built config, with job names checked against the real registry.

    A typo in ``MAINTENANCE_JOBS_DISABLED`` fails at the first read with the
    valid names in the message, instead of silently disabling nothing.
    """
    config = MaintenanceConfig()
    from .registry import DEFAULT_JOBS, validate_jobs_disabled

    validate_jobs_disabled(config.jobs_disabled, DEFAULT_JOBS)
    return config

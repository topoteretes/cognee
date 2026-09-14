"""Configuration owned by the improve loop itself (plan Part 5.9).

``ImproveConfig`` holds only the knobs the loop owns. Shared knobs stay with
their owners and are read where they live: ``triplet_embedding`` is cognify's
(``get_cognify_config()``), ``caching`` / ``auto_feedback`` belong to the cache
layer, ``default_feedback_influence`` and ``personalization_enabled`` to
``base_config``. Re-declaring one of those here is the drift this package
removes.

Environment variables (prefix ``IMPROVE_``)::

    IMPROVE_AUTO_ENABLED=true          # auto-improve after remember()
    IMPROVE_DEBOUNCE_ENTRIES=1         # run only after N new session entries
    IMPROVE_DEBOUNCE_SECONDS=0         # ... or after T seconds
    IMPROVE_STAGES_DISABLED=a,b        # csv of stage names to skip
    IMPROVE_FEEDBACK_ALPHA=0.1         # learning rate for feedback weights, (0, 1]
"""

from functools import lru_cache
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .constants import DEFAULT_FEEDBACK_ALPHA


class ImproveConfig(BaseSettings):
    """Settings for the self-improvement loop (env prefix ``IMPROVE_``)."""

    auto_enabled: bool = True
    # Debounce for automatic improves (plan item B6). The defaults mean "no
    # debounce": every trigger runs, exactly as today. With debounce_seconds
    # set and debounce_entries left at this default 1 — which fires on every
    # call — the entry trigger steps aside, so a seconds-only configuration
    # is time-only rather than a silent no-op (set entries >= 2 to combine).
    debounce_entries: int = 1
    debounce_seconds: float = 0.0
    # Stage names (see ``registry.DEFAULT_STAGES``) to skip with reason
    # ``disabled_by_config``. Read as a comma-separated string.
    stages_disabled: Annotated[list[str], NoDecode] = []
    feedback_alpha: float = DEFAULT_FEEDBACK_ALPHA

    model_config = SettingsConfigDict(env_prefix="IMPROVE_", env_file=".env", extra="ignore")

    @field_validator("stages_disabled", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(part).strip() for part in value if str(part).strip()]

    @field_validator("feedback_alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("feedback_alpha must be in range (0, 1]")
        return value

    @field_validator("debounce_entries")
    @classmethod
    def _entries_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("debounce_entries must be >= 0")
        return value

    @field_validator("debounce_seconds")
    @classmethod
    def _seconds_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("debounce_seconds must be >= 0")
        return value


@lru_cache
def get_improve_config() -> ImproveConfig:
    return ImproveConfig()

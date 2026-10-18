"""Configuration owned by the improve loop itself (plan Part 5.9).

``ImproveConfig`` holds only the knobs the loop owns. Shared knobs stay with
their owners and are read where they live: ``caching`` / ``auto_feedback``
belong to the cache layer, ``default_feedback_influence`` and ``personalization_enabled`` to
``base_config``. Re-declaring one of those here is the drift this package
removes.

Environment variables (prefix ``IMPROVE_``)::

    IMPROVE_AUTO_ENABLED=true          # auto-improve after remember()
    IMPROVE_DEBOUNCE_ENTRIES=1         # run only after N new session entries
    IMPROVE_DEBOUNCE_SECONDS=0         # ... or after T seconds
    IMPROVE_STAGES_DISABLED=a,b        # csv of stage names to skip
    IMPROVE_FEEDBACK_ALPHA=0.1         # learning rate for feedback weights, (0, 1]
    IMPROVE_REVIEW_CONFLICTS=false     # review source facts and dated conflicts
    IMPROVE_EFFECTIVE_DATE_KEY=effective_date  # document metadata source-date key
    IMPROVE_TRACE_WINDOW_STEPS=0       # agent trace steps per persisted document; 0 = all pending
    IMPROVE_TRACE_WINDOWS_PER_RUN=0    # max agent trace documents per run; 0 = drain
    IMPROVE_DISTILL_MAX_BATCHES_PER_RUN=0  # curator batches per distill run; 0 = uncapped
"""

from functools import lru_cache
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from cognee.shared.logging_utils import get_logger

from .constants import DEFAULT_FEEDBACK_ALPHA

logger = get_logger("improve")


class ImproveConfig(BaseSettings):
    """Settings for the self-improvement loop (env prefix ``IMPROVE_``)."""

    auto_enabled: bool = True
    review_conflicts: bool = False
    effective_date_key: str = "effective_date"
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
    # Agent trace persistence cognifies a session's whole pending backlog as one
    # document (chunking bounds the LLM calls, not the document). A value caps
    # the steps per document; the rest follows in later passes of the same run.
    trace_window_steps: int = 0
    # Windows one persist_agent_traces run may cognify. 0 (the default) drains
    # the backlog in one run; with a cap, the rest waits for the next improve().
    trace_windows_per_run: int = 0
    # Session distillation curates every new gated entry of a session per run.
    # A cap curates only the most relevant entries that fit in this many curator
    # batches and seals the rest uncurated; 0 means no cap.
    distill_max_batches_per_run: int = 0

    model_config = SettingsConfigDict(env_prefix="IMPROVE_", extra="ignore")

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

    @field_validator("trace_window_steps")
    @classmethod
    def _window_steps_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trace_window_steps must be >= 0 (0 = all pending steps)")
        return value

    @field_validator("trace_windows_per_run")
    @classmethod
    def _windows_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trace_windows_per_run must be >= 0 (0 = drain the backlog)")
        return value

    @field_validator("debounce_entries")
    @classmethod
    def _entries_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("debounce_entries must be >= 0")
        return value

    @field_validator("distill_max_batches_per_run")
    @classmethod
    def _cap_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("distill_max_batches_per_run must be >= 0 (0 = uncapped)")
        return value

    @field_validator("debounce_seconds")
    @classmethod
    def _seconds_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("debounce_seconds must be >= 0")
        return value


@lru_cache
def get_improve_config() -> ImproveConfig:
    """The env-built config, validated against the real stage registry.

    Stage names are checked HERE, not in ``ImproveConfig`` itself: the model
    must stay constructible with arbitrary names for non-default stage sets
    (tests, custom registries), but an env typo must fail at the first read
    with the valid names in the message — the API server touches this at
    startup so a bad ``IMPROVE_STAGES_DISABLED`` fails the boot, not every
    later request.
    """
    config = ImproveConfig()
    from .registry import DEFAULT_STAGES, REMOVED_STAGE_NAMES, validate_stages_disabled

    validate_stages_disabled(config.stages_disabled, DEFAULT_STAGES)
    # Warned here, behind the cache, so it fires once per process and not on
    # every improve() run, which re-validates.
    removed = sorted(set(config.stages_disabled) & REMOVED_STAGE_NAMES)
    if removed:
        logger.warning("IMPROVE_STAGES_DISABLED names removed stage(s) %s; ignoring them", removed)
    return config

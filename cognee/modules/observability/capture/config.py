"""Configuration for eval capture (SDK-529).

Fields are named after the full environment variables (repo convention — no
``env_prefix``). The ``COGNEE_`` prefix follows ``COGNEE_TRACING_ENABLED``;
unprefixed ``CAPTURE_*`` would be ambiguous.

The base config is read lazily inside the validator: ``cognee/base_config.py``
imports the observability package, so a module-level ``get_base_config`` import
here would create a partially-initialized-module cycle, and a class-time default
would freeze ``DATA_ROOT_DIRECTORY`` and break monkeypatching in tests.
"""

import os
from functools import lru_cache

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


class CaptureConfig(BaseSettings):
    """Knobs for the non-blocking eval capture hook.

    Read once per process by ``hook._ensure_initialized()``; never on the emit
    path. Directory defaults to ``<data_root>/capture`` (not ``evals``, which
    collides with ``cognee/eval_framework`` artifacts).
    """

    cognee_capture_enabled: bool = False
    cognee_capture_dir: str | None = None
    cognee_capture_queue_size: int = 512
    cognee_capture_batch_size: int = 64
    cognee_capture_flush_interval_s: float = 2.0
    cognee_capture_sample_rate: float = 1.0
    # Upper bound on one sink write; a wedged sink (S3 under partition) must not
    # pin the flusher and every later drain().
    cognee_capture_sink_timeout_s: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @pydantic.model_validator(mode="after")
    def fill_derived(self):
        if not 0.0 <= self.cognee_capture_sample_rate <= 1.0:
            raise ValueError(
                "COGNEE_CAPTURE_SAMPLE_RATE must be in [0, 1], "
                f"got {self.cognee_capture_sample_rate}"
            )
        if self.cognee_capture_sink_timeout_s <= 0:
            raise ValueError(
                "COGNEE_CAPTURE_SINK_TIMEOUT_S must be positive, "
                f"got {self.cognee_capture_sink_timeout_s}"
            )

        if not self.cognee_capture_dir:
            # Lazy on purpose — see the module docstring.
            from cognee.base_config import get_base_config

            base_config = get_base_config()
            self.cognee_capture_dir = os.path.join(base_config.data_root_directory, "capture")

        return self

    def to_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        return {
            "cognee_capture_enabled": self.cognee_capture_enabled,
            "cognee_capture_dir": self.cognee_capture_dir,
            "cognee_capture_queue_size": self.cognee_capture_queue_size,
            "cognee_capture_batch_size": self.cognee_capture_batch_size,
            "cognee_capture_flush_interval_s": self.cognee_capture_flush_interval_s,
            "cognee_capture_sample_rate": self.cognee_capture_sample_rate,
            "cognee_capture_sink_timeout_s": self.cognee_capture_sink_timeout_s,
        }


@lru_cache
def get_capture_config() -> CaptureConfig:
    """Return the cached capture configuration.

    Tests that change the environment call ``get_capture_config.cache_clear()``
    (``hook._reset_for_tests()`` does it too).
    """
    return CaptureConfig()

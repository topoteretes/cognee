"""Configuration for the Dataset Queue concurrency limiter.

Environment variables:
    - ``DATASET_QUEUE_ENABLED``: turn the queue on/off (default: ``False``).
    - ``DATABASE_MAX_LRU_CACHE_SIZE``: maximum number of concurrent dataset
      operations permitted when the queue is enabled (default: ``10``).
"""

from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_MAX_LRU_CACHE_SIZE = 10


class DatasetQueueConfig(BaseSettings):
    """Pydantic settings model for the Dataset Queue.

    Attributes:
        dataset_queue_enabled: When ``True`` the queue limits the number of
            concurrent dataset operations (search / run_pipeline_per_dataset)
            to :attr:`database_max_lru_cache_size`. When ``False`` operations
            run without any concurrency limit imposed by the queue.
        database_max_lru_cache_size: Maximum number of concurrent dataset
            operations allowed at once. Normalised to a minimum of ``1``.
    """

    dataset_queue_enabled: bool = False
    database_max_lru_cache_size: int = DEFAULT_MAX_LRU_CACHE_SIZE

    # Intentionally do NOT point ``env_file`` at ``.env`` — the queue is a
    # lightweight infrastructure component and should read only its two
    # dedicated environment variables from ``os.environ``. This also keeps
    # tests that use ``patch.dict(os.environ, ..., clear=True)`` hermetic.
    model_config = SettingsConfigDict(extra="allow")

    @field_validator("database_max_lru_cache_size", mode="before")
    @classmethod
    def _coerce_max_size(cls, value: Any) -> int:
        """Coerce the env value into a valid positive integer.

        Invalid/unparseable values fall back to the default. Values below
        ``1`` are clamped up to ``1`` so the semaphore is always usable.
        """
        if value is None:
            return DEFAULT_MAX_LRU_CACHE_SIZE

        # ``bool`` is a subclass of ``int`` in Python; disallow it explicitly
        # so ``DATABASE_MAX_LRU_CACHE_SIZE=True`` does not silently become 1.
        if isinstance(value, bool):
            return DEFAULT_MAX_LRU_CACHE_SIZE

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return DEFAULT_MAX_LRU_CACHE_SIZE
            try:
                value = int(stripped)
            except (ValueError, TypeError):
                # Accept float-style strings like "5.9" -> 5.
                try:
                    value = int(float(stripped))
                except (ValueError, TypeError):
                    return DEFAULT_MAX_LRU_CACHE_SIZE
        elif isinstance(value, float):
            value = int(value)

        if not isinstance(value, int):
            return DEFAULT_MAX_LRU_CACHE_SIZE

        if value < 1:
            return 1
        return value

    def to_dict(self) -> dict:
        """Return a plain-dict representation of the configuration."""
        return {
            "dataset_queue_enabled": self.dataset_queue_enabled,
            "database_max_lru_cache_size": self.database_max_lru_cache_size,
        }


@lru_cache
def get_dataset_queue_config() -> DatasetQueueConfig:
    """Return the process-wide :class:`DatasetQueueConfig` instance."""
    return DatasetQueueConfig()

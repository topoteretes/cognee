"""Dataset Queue — lightweight concurrency limiter for dataset operations.

This package exposes three primary symbols:

* :func:`dataset_queue` — process-wide singleton accessor. Calling it returns
  the shared :class:`DatasetQueue` instance. Assigning ``dataset_queue._instance = None``
  resets the singleton (used by tests).
* :func:`dataset_queue_limit` — decorator that wraps an async function so that
  concurrent calls respect the queue's limit.
* :class:`DatasetQueueTimeoutError` — raised when acquiring a slot with a
  timeout fails.

Configuration is driven by two environment variables:

* ``DATASET_QUEUE_ENABLED`` — turn the feature on/off (default: ``False``).
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — maximum number of concurrent operations
  when enabled (default: ``10``).

See :mod:`cognee.infrastructure.databases.dataset_queue.config` for the
full configuration schema.
"""

from .config import DatasetQueueConfig, get_dataset_queue_config
from .decorator import dataset_queue_limit
from .queue import DatasetQueue, DatasetQueueTimeoutError, dataset_queue

__all__ = [
    "DatasetQueue",
    "DatasetQueueConfig",
    "DatasetQueueTimeoutError",
    "dataset_queue",
    "dataset_queue_limit",
    "get_dataset_queue_config",
]

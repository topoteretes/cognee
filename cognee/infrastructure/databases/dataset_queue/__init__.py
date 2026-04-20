"""Dataset Queue — lightweight concurrency limiter for dataset operations.

Public symbols:

* :func:`dataset_queue` — process-wide singleton accessor. Calling it returns
  the shared :class:`DatasetQueue` instance. Assigning
  ``dataset_queue._instance = None`` resets the singleton (used by tests).
* :func:`dataset_queue_limit` — decorator that wraps an async function so
  concurrent calls respect the queue's limit.
* :class:`DatasetQueueTimeoutError` — raised when acquiring a slot with a
  timeout fails.
* :func:`get_dataset_queue_settings` / :class:`DatasetQueueSettings` — the
  mock seam used by tests and by callers that want to inspect the current
  effective settings.

Configuration (no config file, just env + a shared constant):

* ``DATASET_QUEUE_ENABLED`` — environment variable. Truthy (``1``/``true``/
  ``yes``/``on``, case-insensitive) turns the queue on. Default: off.
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — the shared constant defined in
  :mod:`cognee.shared.lru_cache` (default: ``128``). Controls the max
  concurrent dataset ops when the queue is enabled.
"""

from .decorator import dataset_queue_limit
from .queue import (
    DatasetQueue,
    DatasetQueueSettings,
    DatasetQueueTimeoutError,
    dataset_queue,
    get_dataset_queue_settings,
)

__all__ = [
    "DatasetQueue",
    "DatasetQueueSettings",
    "DatasetQueueTimeoutError",
    "dataset_queue",
    "dataset_queue_limit",
    "get_dataset_queue_settings",
]

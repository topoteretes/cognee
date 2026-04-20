"""Dataset Queue — lightweight concurrency limiter for dataset operations.

Configuration:

* ``DATASET_QUEUE_ENABLED`` — environment variable. Truthy values turn the
  queue on; anything else (including unset) leaves it off. Default: off.
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — shared constant from
  :mod:`cognee.shared.lru_cache` (default: ``128``).

Public surface:

* :func:`dataset_queue` — singleton accessor. The returned object has one
  method, :meth:`~DatasetQueue.acquire`, an async context manager that
  holds a slot for its body.
"""

from .queue import dataset_queue

__all__ = ["dataset_queue"]

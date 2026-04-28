"""Dataset Queue — lightweight concurrency limiter for dataset operations.

Configuration:

* ``DATASET_QUEUE_ENABLED`` — environment variable. Truthy values turn the
  queue on; anything else (including unset) leaves it off. Default: off.
* ``DATASET_QUEUE_MAX_CONCURRENT`` — environment variable. Integer value for maximum concurrent slots in the queue.
"""

from .queue import dataset_queue

__all__ = ["dataset_queue"]

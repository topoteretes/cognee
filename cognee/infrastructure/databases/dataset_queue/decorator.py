"""Decorator for applying the Dataset Queue concurrency limit.

Example:
    >>> from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit
    >>>
    >>> @dataset_queue_limit
    ... async def search(query_text, **kwargs):
    ...     ...
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, TypeVar

from .queue import dataset_queue


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def dataset_queue_limit(func: F) -> F:
    """Wrap an async function so that it runs through the dataset queue.

    The decorator defers queue lookup until call-time so that:
      * the queue's ``enabled`` state is re-evaluated on every call (honouring
        runtime configuration changes), and
      * tests can reset the queue singleton between runs.

    When the queue is disabled the wrapped function runs with no added
    synchronization (the underlying ``queue.acquire()`` is a no-op context
    manager in that state).

    The wrapper preserves ``__name__``, ``__doc__`` and other metadata via
    :func:`functools.wraps`. Exceptions raised by the wrapped function are
    propagated unchanged; the queue slot is released even on exception.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        queue = dataset_queue()
        async with queue.acquire():
            return await func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]

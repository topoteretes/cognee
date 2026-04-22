"""LRU cache that calls .close() on evicted entries."""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from time import time

logger = logging.getLogger(__name__)


def _close_value(value):
    """Call close() on a value, scheduling it as a task if it returns a coroutine.

    If close() is async and no event loop is running, falls back to
    ``asyncio.run()`` to ensure cleanup is not silently skipped.
    """
    if not hasattr(value, "close"):
        return
    result = value.close()
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(result)
        except RuntimeError:
            try:
                asyncio.run(result)
            except Exception:
                logger.warning(
                    "Failed to run async close() for %s during eviction",
                    type(value).__name__,
                    exc_info=True,
                )


class ClosingLRUCache:
    """Thread-safe LRU cache that calls ``close()`` on evicted values.

    Unlike :func:`functools.lru_cache`, evicted entries are cleaned up
    deterministically — their ``close()`` method (if present) is called
    at the moment of eviction, while all fields are still alive.
    """

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()

    def get_or_create(self, key, factory):
        now = time()

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.last_accessed_at = now
                self._cache.move_to_end(key)
                return entry.value

        value = factory()

        with self._lock:
            # Re-check after releasing lock — another thread may have created it.
            if key in self._cache:
                entry = self._cache[key]
                entry.last_accessed_at = now
                self._cache.move_to_end(key)
                cached_value = entry.value
            else:
                cached_value = None

            if cached_value is not None:
                _close_value(value)
                return cached_value

            if self._maxsize > 0 and len(self._cache) >= self._maxsize:
                _, evicted = self._cache.popitem(last=False)
                _close_value(evicted.value)

            self._cache[key] = _CacheEntry(value=value, last_accessed_at=now)
            return value

    def cache_clear(self):
        """Close and remove all cached entries."""
        with self._lock:
            for entry in self._cache.values():
                _close_value(entry.value)
            self._cache.clear()

    def cache_info(self):
        """Return current size and max size."""
        with self._lock:
            return {"size": len(self._cache), "maxsize": self._maxsize}


@dataclass
class _CacheEntry:
    value: object
    last_accessed_at: float


def closing_lru_cache(maxsize: int = 128):
    """Decorator that caches return values in a :class:`ClosingLRUCache`.

    Drop-in replacement for ``@functools.lru_cache`` that calls ``.close()``
    on values evicted from the cache.

    The decorated function gains ``cache_clear()`` and ``cache_info()``
    attributes, matching the ``lru_cache`` API, as well as a ``__wrapped__``
    attribute pointing to the original function.
    """

    def decorator(fn):
        cache = ClosingLRUCache(maxsize=maxsize)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            return cache.get_or_create(key, lambda: fn(*args, **kwargs))

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator

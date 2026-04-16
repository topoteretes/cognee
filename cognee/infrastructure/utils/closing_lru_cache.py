"""LRU cache that calls ``close()`` on values when they're evicted.

Drop-in replacement for ``functools.lru_cache`` for cases where cached
values hold resources (database engines, file handles, network clients)
that need explicit cleanup. When a value leaves the cache — by LRU
eviction, ``cache_clear()``, or ``cache_evict(*args, **kwargs)`` — its
``close()`` method is invoked if present. Both sync and ``async def``
``close`` implementations are supported.
"""

import asyncio
import threading
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Union

from cognee.shared.logging_utils import get_logger

logger = get_logger("closing_lru_cache")


# Sentinel inserted between args and kwargs in the cache key tuple so that
# ``f(1, 2)`` and ``f(1, x=2)`` hash to distinct keys.
_KWARGS_MARK = object()

# Sentinel returned by ``dict.get`` / ``dict.pop`` to mean "key not present".
# Distinct from ``_KWARGS_MARK`` to avoid overloading one object with two
# unrelated meanings.
_MISSING = object()


def _close_value(value: Any) -> None:
    """Invoke ``close()`` on a cached value if it exposes one.

    Sync ``close`` is called directly. Async ``close`` is scheduled on the
    caller's running event loop when one exists, falling back to
    ``asyncio.run`` only when truly outside any loop:

    - **Outside any loop** (typical sync context, including worker
      threads spawned by ``run_in_executor``): ``asyncio.run(close())``
      drives the coroutine to completion on a fresh loop. Safe because
      no resource on this thread is bound to anything else.
    - **Inside a running loop on the current thread** (eviction
      triggered from inside an ``async def``): the close is scheduled
      on that same loop via ``loop.create_task`` so it runs on the loop
      the resource was opened on, instead of being orphaned onto a new
      one. Blocking with ``future.result()`` would deadlock since the
      current thread *is* the loop's thread, so the close becomes
      fire-and-forget — it completes the next time the loop yields.

    Errors raised synchronously by ``close()`` are logged and swallowed
    so that a failing close on one entry does not break eviction of
    subsequent entries or the caller's flow. Errors inside the
    fire-and-forget task surface via the loop's default exception
    handler.
    """
    close = getattr(value, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if not asyncio.iscoroutine(result):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            # No active loop on this thread — drive the close to
            # completion on a fresh loop.
            asyncio.run(result)
        else:
            # Inside a running loop. Schedule on it so the close runs
            # on the loop that owns the resource. Cannot block here
            # without deadlocking the loop, so this is fire-and-forget.
            loop.create_task(result)
    except Exception as exc:
        logger.warning(
            "Error closing evicted cache value of type %s: %s",
            type(value).__name__,
            exc,
        )


def closing_lru_cache(
    maxsize: Union[int, Callable[..., Any], None] = 128,
    typed: bool = False,
):
    """LRU cache decorator that closes evicted values.

    Mirrors ``functools.lru_cache``: supports the same ``maxsize`` /
    ``typed`` arguments, exposes ``cache_clear()`` and ``cache_info()``,
    and additionally provides ``cache_evict(*args, **kwargs)`` for
    explicit single-key removal. Every value that leaves the cache has
    its ``close()`` method called (sync or async); values without
    ``close`` are dropped silently.

    Args:
        maxsize: Maximum entries before LRU eviction. ``None`` disables
            the size limit (entries only leave via explicit clear/evict).
        typed: If ``True``, arguments of distinct types are cached
            separately (e.g. ``f(3)`` and ``f(3.0)`` distinct keys).

    Usage::

        @closing_lru_cache(maxsize=4)
        def get_engine(url: str) -> Engine: ...

        # Bare decorator form is also supported:
        @closing_lru_cache
        def get_client(host: str) -> Client: ...
    """
    # Bare decorator support: @closing_lru_cache (no parens)
    if callable(maxsize):
        func = maxsize
        return closing_lru_cache(128, False)(func)

    def decorator(func: Callable) -> Callable:
        cache: OrderedDict = OrderedDict()
        lock = threading.RLock()
        stats = {"hits": 0, "misses": 0}

        def _make_key(args: tuple, kwargs: dict) -> tuple:
            key: tuple = args
            if kwargs:
                key += (_KWARGS_MARK,)
                for item in kwargs.items():
                    key += item
            if typed:
                key += tuple(type(a) for a in args)
                if kwargs:
                    key += tuple(type(v) for v in kwargs.values())
            return key

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(args, kwargs)

            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    stats["hits"] += 1
                    return cache[key]
                stats["misses"] += 1

            # Compute outside the lock so concurrent calls with different
            # keys don't serialize on a single slow factory call.
            value = func(*args, **kwargs)
            evicted: list = []

            with lock:
                existing = cache.get(key, _MISSING)
                if existing is _MISSING:
                    cache[key] = value
                    if maxsize is not None:
                        while len(cache) > maxsize:
                            _, old_value = cache.popitem(last=False)
                            evicted.append(old_value)
                else:
                    # Another caller filled the same key while we computed;
                    # discard our value and return the established one so
                    # callers don't get duplicate resources.
                    cache.move_to_end(key)
                    evicted.append(value)
                    value = existing

            for old in evicted:
                _close_value(old)

            return value

        def cache_clear() -> None:
            """Empty the cache, closing every cached value."""
            with lock:
                old_entries = list(cache.values())
                cache.clear()
                stats["hits"] = 0
                stats["misses"] = 0
            for value in old_entries:
                _close_value(value)

        def cache_evict(*args, **kwargs) -> bool:
            """Evict a single entry by its arguments. Returns True if removed."""
            key = _make_key(args, kwargs)
            with lock:
                value = cache.pop(key, _MISSING)
            if value is _MISSING:
                return False
            _close_value(value)
            return True

        def cache_info() -> dict:
            with lock:
                return {
                    "hits": stats["hits"],
                    "misses": stats["misses"],
                    "maxsize": maxsize,
                    "currsize": len(cache),
                }

        wrapper.cache_clear = cache_clear
        wrapper.cache_evict = cache_evict
        wrapper.cache_info = cache_info
        return wrapper

    return decorator

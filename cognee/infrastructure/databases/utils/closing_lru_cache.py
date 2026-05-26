"""LRU cache that closes entries after they leave the cache and caller scope."""

import asyncio
import inspect
import logging
import weakref
from collections import OrderedDict
from functools import wraps
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


class CacheInfo(dict):
    """Cache info mapping with functools.lru_cache-style attributes."""

    @property
    def currsize(self):
        return self["size"]

    @property
    def maxsize(self):
        return self["maxsize"]


# Strong refs for fire-and-forget async close() tasks. ``asyncio.create_task``
# returns a task whose only strong reference is our local variable; without
# anchoring here, Python's gc can collect an in-flight eviction task before
# it completes (and the async close never runs). Tasks remove themselves on
# done, so this set's size tracks currently-pending close operations.
_PENDING_CLOSE_TASKS: set = set()


# Sentinel separating positional from keyword args in cache keys. Mirrors
# ``functools.lru_cache``'s ``_kwd_mark`` so calls like ``fn(("a", 1))``
# and ``fn(a=1)`` map to distinct entries — without it, both would
# produce the key ``("a", 1)`` and collide.
_KW_MARK = object()


def _close_value(value):
    """Call close() on a value, scheduling it as a task if it returns a coroutine.

    If close() is async and no event loop is running, falls back to
    ``asyncio.run()`` to ensure cleanup is not silently skipped.
    """
    if not hasattr(value, "close"):
        return
    try:
        result = value.close()
    except Exception:
        # A raising close() must not abort the surrounding loop (eviction
        # iteration in ``cache_clear`` or a single eviction in
        # ``get_or_create``). Log and keep going — the caller already lost
        # the reference, and any partial cleanup is better than none.
        logger.warning(
            "Failed to close %s during eviction",
            type(value).__name__,
            exc_info=True,
        )
        return
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(result)
            except Exception:
                logger.warning(
                    "Failed to run async close() for %s during eviction",
                    type(value).__name__,
                    exc_info=True,
                )
            return

        task = loop.create_task(result)
        _PENDING_CLOSE_TASKS.add(task)

        def _on_close_done(done_task, _value_type=type(value).__name__):
            # Always drop the strong ref so the task can be collected.
            _PENDING_CLOSE_TASKS.discard(done_task)
            # Retrieve the result so failures surface through the same
            # structured ``logger.warning`` channel as the sync /
            # ``asyncio.run()`` branches above. Without this, an async
            # ``close()`` that raises only surfaces as Python's
            # "Task exception was never retrieved" warning at GC time,
            # which ops greps for ``Failed to run async close()`` would miss.
            try:
                done_task.result()
            except Exception:
                logger.warning(
                    "Failed to run async close() for %s during eviction",
                    _value_type,
                    exc_info=True,
                )

        task.add_done_callback(_on_close_done)


class _LeasedCacheEntry:
    """Own one cached value and close it after cache + returned proxy are gone."""

    def __init__(self, value):
        self.value = value
        self.proxy = None
        self.in_cache = True
        self.close_requested = False
        self.closed = False
        self._lock = Lock()

    def lease(self):
        with self._lock:
            if self.closed:
                raise RuntimeError(f"{type(self.value).__name__} cache entry is already closed")
            if self.proxy is None:
                self.proxy = _LeasedValueProxy(self)
            return self.proxy

    def proxy_released(self):
        value_to_close = None
        with self._lock:
            self.proxy = None
            if self.close_requested and not self.closed:
                self.closed = True
                value_to_close = self.value

        if value_to_close is not None:
            _close_value(value_to_close)

    def detach_from_cache(self):
        value_to_close = None
        proxy_to_drop = None
        with self._lock:
            self.in_cache = False
            self.close_requested = True
            proxy_to_drop = self.proxy
            self.proxy = None
            if proxy_to_drop is None and not self.closed:
                self.closed = True
                value_to_close = self.value

        if value_to_close is not None:
            _close_value(value_to_close)
        # Keep ``proxy_to_drop`` alive until after ``self._lock`` is released.
        # If the cache held the last proxy reference, dropping it can run the
        # weakref finalizer, which re-enters ``proxy_released()``.
        del proxy_to_drop


class _LeasedValueProxy:
    """Proxy that keeps a cache entry alive while callers hold it."""

    __slots__ = ("_entry", "_finalizer", "__weakref__")

    def __init__(self, entry: _LeasedCacheEntry):
        object.__setattr__(self, "_entry", entry)
        object.__setattr__(self, "_finalizer", weakref.finalize(self, entry.proxy_released))

    @property
    def __class__(self):
        return self._entry.value.__class__

    @property
    def __wrapped__(self):
        return self._entry.value

    def __repr__(self):
        return repr(self._entry.value)

    def __getattr__(self, name):
        attr = getattr(self._entry.value, name)

        if not callable(attr):
            return attr

        def call_with_lease(*args, **kwargs):
            result = attr(*args, **kwargs)
            if inspect.isawaitable(result):

                async def await_with_lease(_self=self):
                    # The closure keeps ``self`` alive until the awaitable
                    # completes, which matters for one-liners like
                    # ``await get_vector_engine().search(...)``.
                    return await result

                return await_with_lease()

            return result

        return call_with_lease

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self._entry.value, name, value)


class ClosingLRUCache:
    """Thread-safe LRU cache that closes values after cache and caller use.

    By default, cached values are returned through a stable proxy. Evicting
    or clearing an entry removes it from future cache hits immediately, but
    delays ``close()`` until the previously returned proxy is no longer held
    by caller code. This keeps stale-but-live adapter references usable while
    still cleaning up detached entries once they are genuinely unused.
    """

    def __init__(self, maxsize: Optional[int] = 128, lease: bool = True):
        """``maxsize`` semantics mirror ``functools.lru_cache``:

        - ``int > 0`` — bounded LRU. The least-recently-used entry is evicted
          on insert and its ``close()`` is requested.
        - ``int <= 0`` — cache disabled. ``factory()`` is called on every
          request and the result is returned to the caller without being
          stored. ``close()`` is NOT called: the caller owns the lifecycle,
          just like ``functools.lru_cache(maxsize=0)`` returns a fresh value
          per call.
        - ``None`` — unbounded. Entries are never evicted.

        ``lease=True`` (default) returns a stable proxy per cached entry and
        defers close until cache ownership and caller references are gone.
        ``lease=False`` preserves the old immediate-close-on-eviction mode.
        """
        if isinstance(maxsize, int):
            if maxsize < 0:
                maxsize = 0
        elif maxsize is not None:
            raise TypeError("maxsize must be an int or None")
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lease = lease
        self._lock = Lock()

    def _wrap_cached_value(self, entry):
        if self._lease:
            return entry.lease()
        return entry.value

    def _make_entry(self, value):
        return _LeasedCacheEntry(value)

    def _detach_entry(self, entry):
        if self._lease:
            entry.detach_from_cache()
        else:
            _close_value(entry.value)

    def get_or_create(self, key, factory):
        # Disabled-cache mode: act as a pass-through. Caller owns the value's
        # lifecycle — matches ``functools.lru_cache(maxsize=0)``.
        if self._maxsize == 0:
            return factory()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._wrap_cached_value(self._cache[key])

        value = factory()
        entry = self._make_entry(value)

        # Decide outcome under the lock; defer ``_close_value`` until
        # after release. ``_close_value`` can run arbitrary user code
        # (sync ``close()``, ``asyncio.run`` for an async ``close()``,
        # logging) and even re-enter cache creation in some adapter
        # close paths — running it under ``self._lock`` would either
        # stall every cache user or deadlock outright.
        loser_value = None
        evicted_value = None
        with self._lock:
            # Re-check after releasing lock — another thread may have created it.
            if key in self._cache:
                self._cache.move_to_end(key)
                loser_value = value
                cached = self._wrap_cached_value(self._cache[key])
            else:
                # ``None`` means unbounded — skip the eviction check entirely.
                if self._maxsize is not None and len(self._cache) >= self._maxsize:
                    _, evicted_value = self._cache.popitem(last=False)
                self._cache[key] = entry
                cached = self._wrap_cached_value(entry)

        if loser_value is not None:
            _close_value(loser_value)
        if evicted_value is not None:
            self._detach_entry(evicted_value)
        return cached

    def cache_clear(self):
        """Close and remove all cached entries."""
        # Capture the values under the lock; close them after release so
        # arbitrary close() code can't stall every cache user (or
        # deadlock by re-entering cache creation).
        with self._lock:
            entries = list(self._cache.values())
            self._cache.clear()
        for entry in entries:
            self._detach_entry(entry)

    def evict(self, key) -> bool:
        """Remove a single entry by key and request its close.

        Returns True if an entry was evicted, False if the key wasn't
        cached. Uses the same defer-close-after-lock pattern as
        ``get_or_create`` / ``cache_clear`` so user close() code can't
        deadlock the cache.
        """
        with self._lock:
            entry = self._cache.pop(key, None)
        if entry is None:
            return False
        self._detach_entry(entry)
        return True

    def contains(self, key) -> bool:
        """Check whether *key* is currently in the cache without creating."""
        with self._lock:
            return key in self._cache

    def cache_info(self):
        """Return current size and max size."""
        with self._lock:
            return CacheInfo(size=len(self._cache), maxsize=self._maxsize)


def closing_lru_cache(maxsize: Optional[int] = 128, lease: bool = True):
    """Decorator that caches return values in a :class:`ClosingLRUCache`.

    Drop-in replacement for ``@functools.lru_cache`` that closes values once
    they are both removed from the cache and no longer held by caller code.
    ``maxsize`` semantics match ``functools.lru_cache``: positive int =
    bounded; ``0`` (or negative) = disabled; ``None`` = unbounded.

    The decorated function gains ``cache_clear()`` and ``cache_info()``
    attributes, matching the ``lru_cache`` API, as well as a ``__wrapped__``
    attribute pointing to the original function.
    """

    def decorator(fn):
        cache = ClosingLRUCache(maxsize=maxsize, lease=lease)

        def _key(args, kwargs):
            # ``_KW_MARK`` separates positional from keyword args so
            # ``fn(("a", 1))`` and ``fn(a=1)`` don't collide.
            if kwargs:
                return args + (_KW_MARK,) + tuple(sorted(kwargs.items()))
            return args

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return cache.get_or_create(_key(args, kwargs), lambda: fn(*args, **kwargs))

        def cache_evict(*args, **kwargs) -> bool:
            """Evict a single entry whose key matches the given args.

            Pass exactly the same args the original call used — the key
            is built with the same scheme as ``wrapper``. Returns True
            if the entry was present.
            """
            return cache.evict(_key(args, kwargs))

        def cache_contains(*args, **kwargs) -> bool:
            """Return True if the key is cached, without creating."""
            return cache.contains(_key(args, kwargs))

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_evict = cache_evict
        wrapper.cache_contains = cache_contains
        wrapper.cache_info = cache.cache_info
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator

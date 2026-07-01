"""LRU cache that closes entries after they leave the cache and caller scope."""

import asyncio
import concurrent.futures
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


def _start_close(value) -> Optional[concurrent.futures.Future]:
    """Begin closing a value and return a ``concurrent.futures.Future`` that
    resolves once the close (including any async worker-process teardown) has
    completed — or ``None`` when there is nothing to wait for (no ``close()``,
    a sync ``close()`` that already finished, or a ``close()`` that raised
    synchronously).

    The returned future lets the cache track "this key is still closing" so a
    later async creation for the same key can ``await`` it before opening a new
    DB worker on the same path (see ``ClosingLRUCache._track_close`` /
    ``aget_or_create``). It is a thread-safe ``concurrent.futures.Future`` so an
    async waiter can ``await asyncio.wrap_future(...)`` it and sync callers can
    poll it.

    Because engine resolution is async (see ``get_graph_engine`` /
    ``get_vector_engine``), the async ``close()`` is run as an ordinary task on
    the running loop and the awaiting creator yields the loop so it can run — no
    off-loop thread is needed. Falls back to ``asyncio.run()`` when no loop is
    running (e.g. a GC finalizer firing off-loop, or interpreter teardown).
    """
    if not hasattr(value, "close"):
        return None
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
        return None
    if not asyncio.iscoroutine(result):
        # Sync close() already completed.
        return None

    value_type = type(value).__name__
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: run to completion now (original fallback). Return an
        # already-resolved future so callers see a uniform type and never block
        # on it (it's done before they look).
        cf: concurrent.futures.Future = concurrent.futures.Future()
        try:
            asyncio.run(result)
        except Exception:
            logger.warning(
                "Failed to run async close() for %s during eviction",
                value_type,
                exc_info=True,
            )
        cf.set_result(None)
        return cf

    # Running loop: schedule the close as a task and mirror its completion into a
    # concurrent.futures.Future. A creator awaiting on the same loop via
    # ``asyncio.wrap_future`` yields control so this task can run (no deadlock).
    cf = concurrent.futures.Future()
    task = loop.create_task(result)
    _PENDING_CLOSE_TASKS.add(task)

    def _on_close_done(done_task, _value_type=value_type, _cf=cf):
        # Always drop the strong ref so the task can be collected.
        _PENDING_CLOSE_TASKS.discard(done_task)
        # Retrieve the result so failures surface through the same structured
        # ``logger.warning`` channel as the ``asyncio.run()`` branch. Without
        # this, an async ``close()`` that raises only surfaces as Python's
        # "Task exception was never retrieved" warning at GC time.
        try:
            done_task.result()
        except Exception:
            logger.warning(
                "Failed to run async close() for %s during eviction",
                _value_type,
                exc_info=True,
            )
        # Always resolve the mirror future as done (never propagate the close
        # failure to waiters — a creator should proceed regardless; the worker
        # open-retry covers a still-held lock).
        if not _cf.done():
            _cf.set_result(None)

    task.add_done_callback(_on_close_done)
    return cf


class _LeasedCacheEntry:
    """Own one cached value and close it after cache + returned proxy are gone."""

    def __init__(self, value, key=None, cache=None):
        self.value = value
        # ``key`` + ``cache`` let the deferred close paths (``proxy_released``,
        # ``detach_from_cache``) register the close as in-flight under this
        # entry's cache key, so a later ``aget_or_create`` for the same key can
        # wait for the lock-holding worker to exit. ``cache`` is the owning
        # ``ClosingLRUCache``; ``None`` keeps the entry usable in isolation
        # (tests construct it directly).
        self.key = key
        self.cache = cache
        self.proxy = None
        self.in_cache = True
        self.close_requested = False
        self.closed = False
        self._lock = Lock()

    def _close(self, value):
        """Route a close through the owning cache so it is tracked as in-flight
        for this key; fall back to an untracked close when there is no cache."""
        if self.cache is not None:
            self.cache._track_close(self.key, value)
        else:
            _start_close(value)

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
            self._close(value_to_close)

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
            self._close(value_to_close)
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
        # The proxy's own slots (_entry, _finalizer) are only set via
        # object.__setattr__ in __init__, so every write reaching here targets
        # the wrapped value and must forward to it.
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
        # Keyed registry of in-flight closes. A key is present here from the
        # moment its value's ``close()`` actually starts until that close
        # (including async worker-process teardown) completes. ``aget_or_create``
        # waits on the matching future before constructing a new value, so a new
        # DB worker never opens a file path whose previous worker is still
        # releasing its lock. Guarded by ``self._lock``.
        self._closing: dict = {}

    def _track_close(self, key, value) -> None:
        """Close ``value`` and, if the close is async/in-flight, record it under
        ``key`` so a concurrent ``aget_or_create`` for the same key waits for it.

        A ``None`` / already-resolved future means the close finished
        synchronously, so there is nothing to register.
        """
        cf = _start_close(value)
        if cf is None or cf.done():
            return
        with self._lock:
            self._closing[key] = cf

        def _cleanup(done_future, _key=key):
            with self._lock:
                # Only clear if we are still the registered future — a newer
                # close for the same key may have superseded us.
                if self._closing.get(_key) is done_future:
                    self._closing.pop(_key, None)

        cf.add_done_callback(_cleanup)

    def _wrap_cached_value(self, entry):
        if self._lease:
            return entry.lease()
        return entry.value

    def _make_entry(self, key, value):
        return _LeasedCacheEntry(value, key=key, cache=self)

    def _detach_entry(self, entry):
        if self._lease:
            entry.detach_from_cache()
        else:
            self._track_close(entry.key, entry.value)

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
        entry = self._make_entry(key, value)

        # Decide outcome under the lock; defer ``_track_close`` until
        # after release. The close can run arbitrary user code
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
            self._track_close(key, loser_value)
        if evicted_value is not None:
            self._detach_entry(evicted_value)
        return cached

    async def aget_or_create(self, key, factory):
        """Async counterpart of :meth:`get_or_create` that waits for an in-flight
        close of the same key before constructing a new value.

        This is the point of the closing registry: when an entry was just
        evicted and its DB worker is still shutting down (still holding the
        on-disk file lock), constructing a new value here would spawn a second
        worker that races the first for the lock and fails. Awaiting the close
        future via ``asyncio.wrap_future`` suspends this coroutine without
        blocking the loop, so the close (which runs as a task on this same loop)
        can complete — releasing the lock — before we construct.

        Construction itself still goes through the synchronous
        :meth:`get_or_create` (which re-checks the cache and only calls
        ``factory`` on a miss).
        """
        if self._maxsize == 0:
            return factory()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._wrap_cached_value(self._cache[key])
            pending_close = self._closing.get(key)

        if pending_close is not None and not pending_close.done():
            try:
                await asyncio.wrap_future(pending_close)
            except Exception:
                # The close future never propagates failures (see _start_close);
                # this guard is belt-and-suspenders so a creator always proceeds.
                pass

        return self.get_or_create(key, factory)

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

        async def acall(*args, **kwargs):
            """Async acquisition that waits for an in-flight close of the same
            key before constructing. Pass exactly the args the sync ``wrapper``
            would use so the cache key matches.
            """
            return await cache.aget_or_create(_key(args, kwargs), lambda: fn(*args, **kwargs))

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
        wrapper.acall = acall
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator

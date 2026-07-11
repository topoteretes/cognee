"""LRU cache that closes entries after they leave the cache and caller scope.

Cached values are live database engines (in subprocess mode: a worker process
holding an exclusive file lock), so removal and death are separate events with
an ordered lifecycle::

    cached --(evict / clear / capacity)--> detached --(last proxy drops)--> closing --> closed
       |                                      |                                           |
       |                            pending-close future                          future resolves;
    leased proxies stay usable      registered HERE                               creators proceed

A key is present in the pending-close registry from the moment its entry
leaves the cache — including while the close is still deferred behind a held
caller proxy — until ``close()`` has fully completed (for subprocess adapters:
the worker exited and released its on-disk lock). Creators for the same key
wait on that future so a new engine never races a dying one for the same
resource:

- ``aget_or_create``: always awaits the pending close.
- ``get_or_create`` in a thread with no running event loop: blocks on it.
- ``get_or_create`` on the event loop: cannot block (the close may need this
  very loop to progress); the adapters' open-retry remains the backstop for
  this residual window.

Capacity eviction honors an optional ``pinned_predicate`` (see
``dataset_queue.pinning``): pinned entries are skipped in LRU order, and when
every entry is pinned the cache temporarily overflows ``maxsize`` instead of
closing a value that is still in use — bounded by the dataset queue's slot
count, converging back once pins lift. Explicit eviction (``evict``,
``evict_where``, ``cache_clear``) ignores pins; those are intentional
lifecycle events.
"""

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


# Dedicated threads for closing subprocess-backed adapters off the caller's
# event loop. Such adapters hold an OS file lock via a worker process; a
# *synchronous* re-resolution for the same DB path (e.g. the engine handle's
# ``__getattr__`` path) blocks the event loop while the new worker waits, which
# would prevent a loop-scheduled close from ever running and releasing the lock.
# Running these closes on their own thread frees the lock regardless of loop
# availability. Safe because the subprocess adapters' ``close()`` is written to
# be event-loop-agnostic (no loop-bound locks; all native teardown via
# ``asyncio.to_thread`` / blocking ``session.shutdown``).
_CLOSE_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="closing-lru-close"
)


def _run_close_coro_blocking(coro, value_type) -> None:
    try:
        asyncio.run(coro)
    except Exception:
        logger.warning(
            "Failed to run async close() for %s during eviction",
            value_type,
            exc_info=True,
        )


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
    later creation for the same key can wait for the lock-holding worker to
    exit before opening the DB again (see ``ClosingLRUCache._track_close`` /
    ``aget_or_create``). It is a thread-safe ``concurrent.futures.Future`` so an
    async waiter on any loop can ``await asyncio.wrap_future(...)`` it and sync
    callers can poll it.

    Execution model is deliberately unchanged from the original
    ``_close_value``: an async ``close()`` is scheduled as a task on the
    *currently running* loop (preserving same-loop semantics for adapters whose
    ``close()`` disposes loop-bound resources, e.g. SQLAlchemy async engines),
    and falls back to ``asyncio.run()`` when no loop is running. Only the
    completion signalling is new.
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

    # Subprocess-backed adapters must release their OS file lock independently of
    # the caller's event loop (see ``_CLOSE_THREAD_POOL``). Run their close on a
    # dedicated thread so a synchronous re-resolution that blocks the loop can't
    # wedge the lock-release.
    if getattr(value, "_subprocess_mode", False):
        try:
            return _CLOSE_THREAD_POOL.submit(_run_close_coro_blocking, result, value_type)
        except RuntimeError:
            # The pool rejects new work once the interpreter is shutting down
            # (a proxy finalizer can fire at exit). That's fine: harness's
            # atexit reaper force-terminates worker processes and the OS frees
            # their file locks on exit. Close the coroutine to avoid a spurious
            # "coroutine was never awaited" warning and move on.
            result.close()
            return None

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

    # Running loop: schedule on this loop and mirror completion into a
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
        # entry's cache key, so a later creation for the same key can wait for
        # the lock-holding worker to exit. ``cache`` is the owning
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
            close_deferred = not self.closed

        if close_deferred and self.cache is not None:
            # The close waits for the last caller proxy to be dropped. Register
            # it as pending NOW so a creator arriving in that window waits for
            # this value instead of racing it for the underlying resource
            # (``proxy_released`` -> ``_close`` resolves the future later).
            self.cache._register_pending_close(self.key)
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

    def _leased_entry_active(self):
        """True while this proxy's cache entry is still the live cached value
        (in cache and not pending close). Lets a pinning caller (e.g.
        ``_GraphEngineHandle``) detect that the entry was evicted and re-resolve
        a fresh value instead of holding the lease open — which would keep an
        evicted DB worker alive and block a new worker on the file lock.

        Defined as a real method so normal attribute lookup finds it before the
        ``__getattr__`` forwarding below; the leading underscore + ``_leased``
        prefix makes a collision with a wrapped adapter attribute unlikely.
        """
        entry = self._entry
        return entry.in_cache and not entry.close_requested

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
                    # ``await get_vector_engine_async().search(...)``.
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

    def __init__(self, maxsize: Optional[int] = 128, lease: bool = True, pinned_predicate=None):
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

        ``pinned_predicate`` (key -> bool), when given, protects entries from
        capacity eviction while it returns True — e.g. engines of datasets
        that are actively being processed. Pinned entries are skipped in LRU
        order; when every entry is pinned the cache temporarily overflows
        ``maxsize`` rather than closing a value that is still in use.
        Explicit eviction (``evict``, ``evict_where``, ``cache_clear``)
        ignores pins — those are intentional lifecycle events. The predicate
        runs under the cache lock and must not re-enter the cache.
        """
        if isinstance(maxsize, int):
            if maxsize < 0:
                maxsize = 0
        elif maxsize is not None:
            raise TypeError("maxsize must be an int or None")
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lease = lease
        self._pinned_predicate = pinned_predicate
        self._lock = Lock()
        # Keyed registry of pending closes. A key is present here from the
        # moment its entry leaves the cache (detach/evict) — even while the
        # actual ``close()`` is still deferred behind a held caller proxy —
        # until the close (including async worker-process teardown) completes.
        # Creators wait on the matching future before constructing a new
        # value, so a new DB worker never opens a file path whose previous
        # worker still holds the on-disk lock. Guarded by ``self._lock``.
        self._closing: dict = {}

    def _register_pending_close(self, key) -> concurrent.futures.Future:
        """Record that ``key``'s value is on its way to being closed and return
        the future that resolves once the close has fully completed. Reuses an
        existing unresolved future so overlapping detach paths share one."""
        with self._lock:
            pending = self._closing.get(key)
            if pending is not None and not pending.done():
                return pending
            pending = concurrent.futures.Future()
            self._closing[key] = pending
            return pending

    def _resolve_pending_close(self, key, pending) -> None:
        """Mark ``key``'s pending close as finished and wake any waiters."""
        with self._lock:
            if self._closing.get(key) is pending:
                self._closing.pop(key, None)
        # Resolve outside the lock: done-callbacks run synchronously in the
        # resolving thread and may re-enter the cache.
        if not pending.done():
            pending.set_result(None)

    def _track_close(self, key, value) -> None:
        """Close ``value`` and resolve ``key``'s pending-close future once the
        close (including async worker-process teardown) has fully completed.

        The future is normally pre-registered at detach time; registering here
        as well covers untracked paths (e.g. a lost create race's loser value).
        """
        pending = self._register_pending_close(key)
        cf = _start_close(value)
        if cf is None or cf.done():
            self._resolve_pending_close(key, pending)
            return

        def _on_close_complete(_done_future, _key=key, _pending=pending):
            self._resolve_pending_close(_key, _pending)

        cf.add_done_callback(_on_close_complete)

    async def await_pending_closes(self, predicate=None) -> None:
        """Wait until every pending close whose key satisfies ``predicate``
        (all pending closes when ``None``) has fully completed."""
        with self._lock:
            pendings = [
                future
                for key, future in self._closing.items()
                if (predicate is None or predicate(key)) and not future.done()
            ]
        for future in pendings:
            await asyncio.wrap_future(future)

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
            pending_close = self._closing.get(key)

        if pending_close is not None and not pending_close.done():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No loop in this thread: block until the previous value for
                # this key has fully closed (worker exited, locks released).
                # Closes are bounded by the adapters' own shutdown timeouts.
                pending_close.result()
            # With a running loop we must not block it — the close may be a
            # task scheduled on this very loop. Async callers go through
            # ``aget_or_create``; this residual sync-on-loop window keeps the
            # worker open-retry as its backstop.

        value = factory()
        entry = self._make_entry(key, value)

        # Decide outcome under the lock; defer ``_track_close`` until
        # after release. The close can run arbitrary user code
        # (sync ``close()``, ``asyncio.run`` for an async ``close()``,
        # logging) and even re-enter cache creation in some adapter
        # close paths — running it under ``self._lock`` would either
        # stall every cache user or deadlock outright.
        loser_value = None
        evicted_values = []
        with self._lock:
            # Re-check after releasing lock — another thread may have created it.
            if key in self._cache:
                self._cache.move_to_end(key)
                loser_value = value
                cached = self._wrap_cached_value(self._cache[key])
            else:
                # ``None`` means unbounded — skip the eviction check entirely.
                while self._maxsize is not None and len(self._cache) >= self._maxsize:
                    # Evict the least-recently-used entry that is not pinned.
                    # When every entry is pinned, overflow ``maxsize`` instead
                    # of closing a value that is still actively in use.
                    eviction_key = next(
                        (
                            candidate
                            for candidate in self._cache
                            if self._pinned_predicate is None
                            or not self._pinned_predicate(candidate)
                        ),
                        None,
                    )
                    if eviction_key is None:
                        break
                    evicted_values.append(self._cache.pop(eviction_key))
                self._cache[key] = entry
                cached = self._wrap_cached_value(entry)

        if loser_value is not None:
            self._track_close(key, loser_value)
        for evicted_value in evicted_values:
            self._detach_entry(evicted_value)
        return cached

    async def aget_or_create(self, key, factory):
        """Async counterpart of :meth:`get_or_create` that waits for an in-flight
        close of the same key before constructing a new value.

        The wait is the whole point of the closing registry: when an entry was
        just evicted and its DB worker is still shutting down (and so still holds
        the on-disk file lock), constructing a new value here would spawn a
        second worker that races the first for the lock and fails. Awaiting the
        close future via ``asyncio.wrap_future`` suspends this coroutine without
        blocking the loop, so the close (which may run as a task on this same
        loop) can complete first.

        The actual construction still goes through the synchronous
        :meth:`get_or_create` (which re-checks the cache and only calls
        ``factory`` on a miss) — the residual window between the await and that
        call is covered by the worker's open-retry backstop.
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

    def evict_where(self, predicate) -> int:
        """Remove every entry whose key satisfies *predicate* and request close.

        Returns the number of entries evicted. Uses the same
        defer-close-after-lock pattern as ``evict``. The predicate runs under
        the cache lock, so it must be a pure function of the key.
        """
        with self._lock:
            matched_keys = [key for key in self._cache if predicate(key)]
            entries = [self._cache.pop(key) for key in matched_keys]
        for entry in entries:
            self._detach_entry(entry)
        return len(entries)

    def contains(self, key) -> bool:
        """Check whether *key* is currently in the cache without creating."""
        with self._lock:
            return key in self._cache

    def cache_info(self):
        """Return current size and max size."""
        with self._lock:
            return CacheInfo(size=len(self._cache), maxsize=self._maxsize)


def closing_lru_cache(maxsize: Optional[int] = 128, lease: bool = True, pinned_predicate=None):
    """Decorator that caches return values in a :class:`ClosingLRUCache`.

    Drop-in replacement for ``@functools.lru_cache`` that closes values once
    they are both removed from the cache and no longer held by caller code.
    ``maxsize`` semantics match ``functools.lru_cache``: positive int =
    bounded; ``0`` (or negative) = disabled; ``None`` = unbounded.
    ``pinned_predicate`` protects matching keys from capacity eviction (see
    :class:`ClosingLRUCache`). A predicate that exposes ``bind_signature`` is
    bound to the cached function's parameter-name -> position map at
    decoration time, so it can address key fields by parameter name and fail
    loudly on a name that doesn't exist (e.g.
    ``dataset_queue_pin_predicate("graph_database_name")``).

    The decorated function gains ``cache_clear()`` and ``cache_info()``
    attributes, matching the ``lru_cache`` API, as well as a ``__wrapped__``
    attribute pointing to the original function.
    """

    def decorator(fn):
        # Parameter-name -> positional index of ``fn``, so key-scanning helpers
        # (``cache_evict_matching``) and name-bound pin predicates can resolve
        # named criteria against the positional part of a cache key.
        _param_positions = {
            name: index for index, name in enumerate(inspect.signature(fn).parameters)
        }

        bind_signature = getattr(pinned_predicate, "bind_signature", None)
        if callable(bind_signature):
            bind_signature(_param_positions)

        cache = ClosingLRUCache(maxsize=maxsize, lease=lease, pinned_predicate=pinned_predicate)

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

        def cache_evict_where(predicate) -> int:
            """Evict every cached entry whose key satisfies *predicate*.

            The predicate receives the raw cache key: the positional-args
            tuple, extended with ``(_KW_MARK, *sorted kwarg items)`` when the
            cached call used keyword arguments. Returns the number of
            evicted entries.
            """
            return cache.evict_where(predicate)

        def cache_evict_matching(**criteria) -> int:
            """Evict every cached entry created with ALL the given argument values.

            Criteria are matched by *parameter name* against the cached call's
            arguments, whether they were passed positionally or by keyword —
            e.g. ``cache_evict_matching(graph_database_name="<uuid>")`` evicts
            only entries whose ``graph_database_name`` argument equals that
            value, never entries where the value happens to appear in some
            other field. Callers evicting by a shared identity field (e.g. a
            database name cached under several differently-keyed entries) use
            this instead of reconstructing exact keys.

            Raises ``ValueError`` on no criteria (it would evict the whole
            cache) or on parameter names not in the wrapped function's
            signature (typo protection). Entries whose key does not record a
            criterion's parameter (the call omitted an optional argument) do
            not match. Returns the number of evicted entries.
            """
            return cache.evict_where(_build_matcher(criteria))

        def _build_matcher(criteria):
            if not criteria:
                raise ValueError("cache_evict_matching requires at least one criterion")
            unknown = set(criteria) - set(_param_positions)
            if unknown:
                raise ValueError(f"Unknown parameter name(s) for {fn.__name__}: {sorted(unknown)}")

            def matches(key) -> bool:
                if _KW_MARK in key:
                    marker_index = key.index(_KW_MARK)
                    positional = key[:marker_index]
                    kwarg_values = dict(key[marker_index + 1 :])
                else:
                    positional = key
                    kwarg_values = {}
                for name, expected in criteria.items():
                    if name in kwarg_values:
                        actual = kwarg_values[name]
                    else:
                        position = _param_positions[name]
                        if position >= len(positional):
                            return False
                        actual = positional[position]
                    if actual != expected:
                        return False
                return True

            return matches

        async def cache_await_closed(**criteria) -> None:
            """Wait until every pending close whose cached call matches
            ``criteria`` has fully completed (workers exited, locks released).
            With no criteria, waits for every pending close in this cache.
            """
            predicate = _build_matcher(criteria) if criteria else None
            await cache.await_pending_closes(predicate)

        def cache_contains(*args, **kwargs) -> bool:
            """Return True if the key is cached, without creating."""
            return cache.contains(_key(args, kwargs))

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_evict = cache_evict
        wrapper.cache_evict_where = cache_evict_where
        wrapper.cache_evict_matching = cache_evict_matching
        wrapper.cache_await_closed = cache_await_closed
        wrapper.cache_contains = cache_contains
        wrapper.cache_info = cache.cache_info
        wrapper.acall = acall
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator

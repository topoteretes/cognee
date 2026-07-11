"""Tests for ClosingLRUCache and the @closing_lru_cache decorator."""

import gc

from cognee.infrastructure.databases.utils.closing_lru_cache import (
    ClosingLRUCache,
    _start_close,
    closing_lru_cache,
)


class _Closeable:
    """Stub with a sync close() that records calls."""

    def __init__(self, name=""):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class _AsyncCloseable:
    """Stub with an async close()."""

    def __init__(self, name=""):
        self.name = name
        self.closed = False

    async def close(self):
        self.closed = True


class _NotCloseable:
    """Stub without a close() method."""

    def __init__(self, name=""):
        self.name = name


class _AsyncWorker(_Closeable):
    async def ping(self):
        import asyncio

        await asyncio.sleep(0)
        return self.closed


class _SlowSubprocessWorker:
    """Stub modeling a subprocess-backed adapter: it advertises
    ``_subprocess_mode`` (so its close runs off-loop) and its async close is
    gated so a test can observe it mid-flight, like tearing down a worker
    process / releasing a file lock.

    When ``started``/``release`` events are supplied, ``close`` sets ``started``
    the instant it begins and then blocks until ``release`` is set, instead of
    sleeping a fixed amount. This removes the wall-clock race where a slow
    ``gc.collect()`` on a loaded CI runner let the fixed-sleep close finish
    before the test could assert it was in flight (``assert True is False``).
    Without the events the close completes immediately.
    """

    def __init__(self, name="", started=None, release=None):
        self.name = name
        self.closed = False
        self._subprocess_mode = True
        self._started = started
        self._release = release

    async def close(self):
        import asyncio

        if self._started is not None:
            self._started.set()
        if self._release is not None:
            # Block a worker thread (not the off-loop) until the test releases
            # us. The 30s timeout is only a safety net so a buggy test that
            # never releases fails instead of hanging CI forever.
            await asyncio.get_running_loop().run_in_executor(None, self._release.wait, 30)
        self.closed = True


class _ThreadRecordingWorker:
    """Records the thread name its async close ran on. Used to assert the
    loop-vs-off-loop branch in ``_start_close``: subprocess-backed values
    (``_subprocess_mode``) close on a pool thread; everything else closes on the
    running loop's thread.
    """

    def __init__(self, subprocess_mode):
        self.closed_on = None
        if subprocess_mode:
            self._subprocess_mode = True

    async def close(self):
        import threading

        self.closed_on = threading.current_thread().name


async def _wait_until(predicate, timeout=5.0):
    """Poll ``predicate`` on the running loop until it is truthy or ``timeout``
    seconds elapse. Used instead of a fixed sleep so assertions about off-loop
    close progress don't depend on wall-clock timing (which flakes on loaded
    CI runners). Returns the final predicate value.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


# -- ClosingLRUCache: basic caching -----------------------------------------


def test_cache_hit_returns_same_object():
    """Repeated get_or_create with the same key returns the cached object."""
    cache = ClosingLRUCache(maxsize=4)
    obj = _Closeable("a")
    result1 = cache.get_or_create("k", lambda: obj)
    result2 = cache.get_or_create("k", lambda: _Closeable("b"))
    assert result1 is result2
    assert result1.name == "a"


def test_different_keys_return_different_values():
    """Different keys produce distinct cached values."""
    cache = ClosingLRUCache(maxsize=4)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    b = cache.get_or_create("b", lambda: _Closeable("b"))
    assert a is not b


def test_cache_info_reports_size_and_maxsize():
    """cache_info returns current size and configured maxsize."""
    cache = ClosingLRUCache(maxsize=5)
    cache.get_or_create("a", lambda: _Closeable())
    cache.get_or_create("b", lambda: _Closeable())
    info = cache.cache_info()
    assert info == {"size": 2, "maxsize": 5}


# -- ClosingLRUCache: eviction -----------------------------------------------


def test_evicts_oldest_and_calls_close():
    """Evicted leased entries close after the returned proxy is released."""
    cache = ClosingLRUCache(maxsize=2)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    a_raw = a.__wrapped__
    cache.get_or_create("b", lambda: _Closeable("b"))
    cache.get_or_create("c", lambda: _Closeable("c"))

    assert a.closed is False
    assert cache.cache_info()["size"] == 2
    del a
    gc.collect()
    assert a_raw.closed is True


def test_access_refreshes_lru_order():
    """Accessing a key moves it to most-recently-used position."""
    cache = ClosingLRUCache(maxsize=2)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    b = cache.get_or_create("b", lambda: _Closeable("b"))
    b_raw = b.__wrapped__

    # Access "a" to refresh it
    cache.get_or_create("a", lambda: _Closeable("should not create"))

    # "c" evicts "b" (now oldest), not "a"
    cache.get_or_create("c", lambda: _Closeable("c"))

    assert a.closed is False
    assert b.closed is False
    del b
    gc.collect()
    assert b_raw.closed is True


def test_eviction_skips_objects_without_close():
    """Evicting an object without close() does not raise."""
    cache = ClosingLRUCache(maxsize=1)
    cache.get_or_create("a", lambda: _NotCloseable("a"))
    cache.get_or_create("b", lambda: _Closeable("b"))


def test_eviction_handles_async_close():
    """Evicting an object with async close() runs it via asyncio.run fallback."""
    cache = ClosingLRUCache(maxsize=1)
    obj = _AsyncCloseable("a")
    cache.get_or_create("a", lambda: obj)
    cache.get_or_create("b", lambda: _Closeable("b"))

    assert obj.closed is True


def test_eviction_async_close_under_running_loop_completes():
    """When a loop is already running, eviction schedules the close coroutine
    on that loop via ``create_task``. The task must be anchored somewhere so
    gc can't collect it before it runs — verify the close actually completes
    before the loop exits.
    """
    import asyncio

    cache = ClosingLRUCache(maxsize=1)

    async def run():
        first = _AsyncCloseable("first")
        cache.get_or_create("a", lambda: first)
        # Triggers eviction of ``first`` while a loop is running.
        cache.get_or_create("b", lambda: _Closeable("b"))
        # Yield twice so the scheduled task has a chance to run.
        for _ in range(3):
            await asyncio.sleep(0)
        return first

    first = asyncio.run(run())
    assert first.closed is True, (
        "async close() scheduled under a running loop must actually execute "
        "— the task was garbage-collected before completing"
    )


def test_eviction_async_close_failure_is_logged_under_running_loop(caplog):
    """When async ``close()`` raises under a running loop, the failure must
    surface through the same ``logger.warning`` channel as the sync /
    ``asyncio.run()`` paths — not just as Python's "Task exception was never
    retrieved" warning at task GC time.
    """
    import asyncio
    import logging

    class _RaisingAsyncCloseable:
        async def close(self):
            raise RuntimeError("boom in async close")

    cache = ClosingLRUCache(maxsize=1)

    async def run():
        cache.get_or_create("a", lambda: _RaisingAsyncCloseable())
        # Triggers eviction of the raising entry while a loop is running.
        cache.get_or_create("b", lambda: _Closeable("b"))
        # Yield so the scheduled task runs to completion.
        for _ in range(3):
            await asyncio.sleep(0)

    with caplog.at_level(
        logging.WARNING,
        logger="cognee.infrastructure.databases.utils.closing_lru_cache",
    ):
        asyncio.run(run())

    matching = [
        r
        for r in caplog.records
        if "Failed to run async close()" in r.getMessage()
        and "_RaisingAsyncCloseable" in r.getMessage()
    ]
    assert matching, (
        "async close() exception under running loop must be logged via "
        "logger.warning, but no matching record was captured"
    )
    # The structured log should carry exc_info for ops grep-ability.
    assert matching[0].exc_info is not None


# -- ClosingLRUCache: cache_clear --------------------------------------------


def test_cache_clear_closes_all_entries():
    """cache_clear detaches entries and closes them when returned proxies are gone."""
    cache = ClosingLRUCache(maxsize=4)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    b = cache.get_or_create("b", lambda: _Closeable("b"))
    c = cache.get_or_create("c", lambda: _Closeable("c"))
    raw_values = [a.__wrapped__, b.__wrapped__, c.__wrapped__]

    cache.cache_clear()

    assert a.closed is False
    assert b.closed is False
    assert c.closed is False
    assert cache.cache_info()["size"] == 0
    del a, b, c
    gc.collect()
    assert all(value.closed for value in raw_values)


def test_cache_clear_skips_objects_without_close():
    """cache_clear does not raise for objects without close()."""
    cache = ClosingLRUCache(maxsize=4)
    cache.get_or_create("a", lambda: _NotCloseable("a"))
    cache.get_or_create("b", lambda: _Closeable("b"))

    cache.cache_clear()
    assert cache.cache_info()["size"] == 0


def test_cache_clear_handles_async_close():
    """cache_clear runs async close() via asyncio.run fallback."""
    cache = ClosingLRUCache(maxsize=4)
    obj = _AsyncCloseable("a")
    cache.get_or_create("a", lambda: obj)

    cache.cache_clear()
    assert obj.closed is True


def test_awaitable_method_holds_lease_after_cache_clear():
    """A one-liner method call keeps the proxy alive until the await completes."""
    import asyncio

    cache = ClosingLRUCache(maxsize=4)

    async def run():
        proxy = cache.get_or_create("a", lambda: _AsyncWorker("a"))
        raw = proxy.__wrapped__
        pending_ping = proxy.ping()
        del proxy
        cache.cache_clear()
        assert raw.closed is False
        closed_during_call = await pending_ping
        assert closed_during_call is False
        gc.collect()
        assert raw.closed is True

    asyncio.run(run())


# -- closing registry: wait-for-in-flight-close ------------------------------


def test_leased_entry_active_reflects_eviction():
    """``_leased_entry_active`` is True while the proxy is the live cache entry
    and False once the entry is evicted — the signal a pinning caller uses to
    drop a stale pin and re-resolve instead of holding an evicted worker alive.
    """
    cache = ClosingLRUCache(maxsize=4)
    proxy = cache.get_or_create("k", lambda: _Closeable("a"))
    assert proxy._leased_entry_active() is True
    cache.evict("k")
    assert proxy._leased_entry_active() is False


def test_subprocess_close_runs_off_loop_and_is_tracked():
    """A value advertising ``_subprocess_mode`` closes off the running loop, and
    the close is registered in the cache's closing registry until it finishes.
    """
    import asyncio
    import threading

    cache = ClosingLRUCache(maxsize=4)

    async def run():
        started = threading.Event()
        release = threading.Event()
        proxy = cache.get_or_create("k", lambda: _SlowSubprocessWorker("first", started, release))
        raw = proxy.__wrapped__
        cache.evict("k")
        del proxy
        gc.collect()
        # Wait until the close has actually begun off-loop (deterministic). It is
        # gated on ``release``, so it cannot have finished — and because the loop
        # is NOT blocked, the registry entry is visible.
        assert await _wait_until(started.is_set)
        assert raw.closed is False
        assert "k" in cache._closing
        # Release it, then confirm it finishes and the registry self-cleans.
        release.set()
        assert await _wait_until(lambda: raw.closed and "k" not in cache._closing)
        assert raw.closed is True
        assert "k" not in cache._closing

    asyncio.run(run())


def test_aget_or_create_waits_for_in_flight_close():
    """A new ``aget_or_create`` for a key whose previous value is still closing
    must wait for that close to finish before constructing the replacement —
    this is what prevents a new DB worker from racing a still-shutting-down one
    for the file lock.
    """
    import asyncio
    import threading

    cache = ClosingLRUCache(maxsize=4)

    async def run():
        started = threading.Event()
        release = threading.Event()
        proxy = cache.get_or_create("k", lambda: _SlowSubprocessWorker("first", started, release))
        raw_first = proxy.__wrapped__
        cache.evict("k")
        del proxy
        gc.collect()
        assert await _wait_until(started.is_set)
        assert raw_first.closed is False  # gated -> definitely in flight

        # ``aget_or_create`` must not return until the in-flight close finishes.
        # Release the close only after a short delay, so aget_or_create is forced
        # to wait for it (deterministically, without a wall-clock in-flight race).
        async def _release_soon():
            await asyncio.sleep(0.05)
            release.set()

        _, second = await asyncio.gather(
            _release_soon(),
            cache.aget_or_create("k", lambda: _SlowSubprocessWorker("second")),
        )
        assert raw_first.closed is True
        assert second.__wrapped__.name == "second"

    asyncio.run(run())


def test_aget_or_create_cache_hit_is_fast_path():
    """``aget_or_create`` returns the cached value without constructing when the
    key is present (and never blocks on the closing registry for a live entry).
    """
    import asyncio

    cache = ClosingLRUCache(maxsize=4)

    async def run():
        first = cache.get_or_create("k", lambda: _Closeable("first"))
        second = await cache.aget_or_create("k", lambda: _Closeable("second"))
        assert first is second
        assert second.name == "first"

    asyncio.run(run())


def test_aget_or_create_concurrent_callers_single_construction():
    """Many coroutines racing ``aget_or_create`` for the same just-evicted key
    must all wait on the one in-flight close and then resolve to a single newly
    constructed value — no thundering-herd of duplicate workers."""
    import asyncio
    import threading

    cache = ClosingLRUCache(maxsize=4)

    async def run():
        started = threading.Event()
        release = threading.Event()
        proxy = cache.get_or_create("k", lambda: _SlowSubprocessWorker("first", started, release))
        raw_first = proxy.__wrapped__
        cache.evict("k")
        del proxy
        gc.collect()
        assert await _wait_until(started.is_set)
        assert raw_first.closed is False  # gated -> definitely in flight

        constructed = []

        def factory():
            w = _SlowSubprocessWorker("new")
            constructed.append(w)
            return w

        async def _release_soon():
            await asyncio.sleep(0.05)
            release.set()

        release_task = asyncio.ensure_future(_release_soon())
        results = await asyncio.gather(*[cache.aget_or_create("k", factory) for _ in range(10)])
        await release_task

        assert len(constructed) == 1, "exactly one new value should be constructed"
        assert all(r.__wrapped__ is results[0].__wrapped__ for r in results)
        assert raw_first.closed is True  # old close completed before construction

    asyncio.run(run())


def test_start_close_subprocess_runs_off_loop_others_on_loop():
    """``_start_close`` runs a ``_subprocess_mode`` value's close on a dedicated
    pool thread (so a loop-blocking sync re-resolution can't wedge lock release)
    while a plain async-close value (e.g. a SQLAlchemy-backed adapter) stays on
    the running loop — preserving loop-bound resource semantics."""
    import asyncio
    import threading

    async def run():
        loop_thread = threading.current_thread().name

        sub = _ThreadRecordingWorker(subprocess_mode=True)
        fut = _start_close(sub)
        assert fut is not None
        await asyncio.wrap_future(fut)
        assert sub.closed_on is not None
        assert sub.closed_on.startswith("closing-lru-close"), (
            f"subprocess close ran on {sub.closed_on}, expected a pool thread"
        )

        normal = _ThreadRecordingWorker(subprocess_mode=False)
        _start_close(normal)
        for _ in range(3):
            await asyncio.sleep(0)
        assert normal.closed_on == loop_thread, (
            f"non-subprocess close ran on {normal.closed_on}, expected the loop thread"
        )

    asyncio.run(run())


def test_decorator_acall_shares_key_with_sync_call():
    """``acall`` builds the same cache key as the sync wrapper, so an async
    acquisition and a sync call for the same args resolve to one object.
    """
    import asyncio

    @closing_lru_cache(maxsize=4)
    def create(key):
        return _Closeable(key)

    async def run():
        a = await create.acall("x")
        b = create("x")
        assert a is b

    asyncio.run(run())


# -- @closing_lru_cache decorator -------------------------------------------


def test_decorator_caches_return_value():
    """Decorated function returns the same object for the same arguments."""
    call_count = 0

    @closing_lru_cache(maxsize=4)
    def create(key):
        nonlocal call_count
        call_count += 1
        return _Closeable(key)

    r1 = create("x")
    r2 = create("x")
    assert r1 is r2
    assert call_count == 1


def test_decorator_different_args_create_different_entries():
    """Different arguments produce distinct cached values."""

    @closing_lru_cache(maxsize=4)
    def create(key):
        return _Closeable(key)

    a = create("a")
    b = create("b")
    assert a is not b
    assert a.name == "a"
    assert b.name == "b"


def test_decorator_evicts_and_closes():
    """Decorated function closes evicted values after returned proxy release."""

    @closing_lru_cache(maxsize=2)
    def create(key):
        return _Closeable(key)

    a = create("a")
    a_raw = a.__wrapped__
    create("b")
    create("c")

    assert a.closed is False
    del a
    gc.collect()
    assert a_raw.closed is True


def test_decorator_exposes_cache_clear():
    """Decorated function has a cache_clear method that detaches all entries."""

    @closing_lru_cache(maxsize=4)
    def create(key):
        return _Closeable(key)

    a = create("a")
    a_raw = a.__wrapped__
    create.cache_clear()
    assert a.closed is False
    del a
    gc.collect()
    assert a_raw.closed is True


def test_decorator_exposes_cache_info():
    """Decorated function has a cache_info method."""

    @closing_lru_cache(maxsize=10)
    def create(key):
        return _Closeable(key)

    create("a")
    create("b")
    info = create.cache_info()
    assert info == {"size": 2, "maxsize": 10}


def test_decorator_exposes_wrapped():
    """Decorated function has __wrapped__ pointing to the original."""

    def original(key):
        return _Closeable(key)

    decorated = closing_lru_cache(maxsize=4)(original)
    assert decorated.__wrapped__ is original


def test_decorator_kwargs_are_part_of_cache_key():
    """Different keyword arguments produce separate cache entries."""

    @closing_lru_cache(maxsize=4)
    def create(a, b="default"):
        return _Closeable(f"{a}-{b}")

    r1 = create("x", b="1")
    r2 = create("x", b="2")
    r3 = create("x", b="1")

    assert r1 is not r2
    assert r1 is r3


# -- maxsize semantics: parity with functools.lru_cache ---------------------


def test_maxsize_zero_disables_cache():
    """maxsize=0 — like functools.lru_cache(maxsize=0): factory runs every
    call, nothing is stored, close() is NOT called (caller owns the value)."""
    import pytest

    cache = ClosingLRUCache(maxsize=0)
    a = _Closeable("a")
    b = _Closeable("a-second")

    r1 = cache.get_or_create("k", lambda: a)
    r2 = cache.get_or_create("k", lambda: b)

    assert r1 is a
    assert r2 is b
    assert a.closed is False, "disabled mode must not close caller-owned values"
    assert b.closed is False
    assert cache.cache_info()["size"] == 0


def test_maxsize_negative_clamped_to_zero():
    """Negative maxsize behaves like maxsize=0 (parity with lru_cache)."""
    cache = ClosingLRUCache(maxsize=-5)
    assert cache.cache_info()["maxsize"] == 0
    obj = _Closeable("a")
    result = cache.get_or_create("k", lambda: obj)
    assert result is obj
    assert obj.closed is False
    assert cache.cache_info()["size"] == 0


def test_maxsize_none_is_unbounded():
    """maxsize=None — never evicts, like functools.lru_cache(maxsize=None)."""
    cache = ClosingLRUCache(maxsize=None)
    objs = [_Closeable(str(i)) for i in range(50)]
    for i, obj in enumerate(objs):
        cache.get_or_create(i, lambda obj=obj: obj)

    # No evictions should have happened.
    for obj in objs:
        assert obj.closed is False
    assert cache.cache_info()["size"] == 50


def test_invalid_maxsize_type_raises():
    """Non-int, non-None maxsize raises TypeError on construction."""
    import pytest

    with pytest.raises(TypeError):
        ClosingLRUCache(maxsize="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ClosingLRUCache(maxsize=1.5)  # type: ignore[arg-type]


def test_decorator_maxsize_zero_disables_cache():
    """The decorator honors maxsize=0 the same way the class does."""
    call_count = 0

    @closing_lru_cache(maxsize=0)
    def create(key):
        nonlocal call_count
        call_count += 1
        return _Closeable(key)

    r1 = create("x")
    r2 = create("x")
    assert r1 is not r2, "maxsize=0 must not cache"
    assert call_count == 2


def test_decorator_maxsize_none_is_unbounded():
    """The decorator honors maxsize=None — every distinct key is retained."""

    @closing_lru_cache(maxsize=None)
    def create(key):
        return _Closeable(key)

    objs = [create(i) for i in range(20)]
    # Re-fetching any earlier key returns the same object — nothing was evicted.
    for i, obj in enumerate(objs):
        assert create(i) is obj


def test_decorator_args_and_kwargs_with_same_payload_do_not_collide():
    """Mirror of ``functools.lru_cache``'s positional/kwargs separation:
    ``fn(("a", 1))`` and ``fn(a=1)`` must produce distinct cache entries
    even though their concatenated tuple representations would otherwise
    coincide. Without the ``_KW_MARK`` sentinel between args and sorted
    kwargs items, both calls would compute key ``("a", 1)`` and the
    cache would return the wrong instance for one of them.
    """
    call_count = 0

    @closing_lru_cache(maxsize=4)
    def create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _Closeable(f"args={args}|kwargs={kwargs}")

    a = create(("a", 1))
    b = create(a=1)
    assert a is not b, "positional and keyword call shapes must not collide"
    assert call_count == 2


def test_cache_evict_matching_by_parameter_name():
    """cache_evict_matching evicts only entries whose named argument equals the
    criterion value, across positional and keyword call shapes."""

    @closing_lru_cache(maxsize=8, lease=False)
    def create(provider, db_name, password=""):
        return _Closeable(f"{provider}/{db_name}")

    kept = create("postgres", "other-db", "secret")
    create("postgres", "dataset-uuid", "secret")
    create("ladybug", db_name="dataset-uuid")

    evicted = create.cache_evict_matching(db_name="dataset-uuid")
    assert evicted == 2

    still_cached = create("postgres", "other-db", "secret")
    assert still_cached is kept, "non-matching entry must survive"


def test_cache_evict_matching_requires_all_criteria():
    """Multiple criteria AND together; a value colliding in a different field
    does not match."""

    @closing_lru_cache(maxsize=8, lease=False)
    def create(provider, db_name):
        return _Closeable(f"{provider}/{db_name}")

    create("postgres", "dataset-uuid")
    # same value, different field: must NOT be touched by db_name criterion
    create("dataset-uuid", "postgres")

    assert create.cache_evict_matching(provider="postgres", db_name="dataset-uuid") == 1
    assert create.cache_info().currsize == 1


def test_cache_evict_matching_rejects_bad_criteria():
    """No criteria (would evict everything) and unknown parameter names raise."""
    import pytest

    @closing_lru_cache(maxsize=8, lease=False)
    def create(provider, db_name):
        return _Closeable(f"{provider}/{db_name}")

    create("postgres", "dataset-uuid")

    with pytest.raises(ValueError):
        create.cache_evict_matching()
    with pytest.raises(ValueError):
        create.cache_evict_matching(db_nmae="typo")
    assert create.cache_info().currsize == 1


def test_pinned_entries_skipped_by_capacity_eviction():
    """Capacity eviction picks the least-recently-used UNPINNED entry."""
    pinned = {("a",)}

    @closing_lru_cache(maxsize=2, lease=False, pinned_predicate=lambda key: key in pinned)
    def create(name):
        return _Closeable(name)

    value_a = create("a")
    value_b = create("b")
    create("c")  # capacity hit: "a" is pinned, so "b" (next LRU) is evicted

    assert value_a.closed is False
    assert value_b.closed is True
    assert create.cache_contains("a")
    assert not create.cache_contains("b")
    assert create.cache_contains("c")


def test_all_pinned_overflows_instead_of_closing():
    """When every entry is pinned the cache exceeds maxsize; nothing closes."""

    @closing_lru_cache(maxsize=2, lease=False, pinned_predicate=lambda key: True)
    def create(name):
        return _Closeable(name)

    values = [create(name) for name in ("a", "b", "c")]

    assert create.cache_info().currsize == 3
    assert all(value.closed is False for value in values)


def test_explicit_evict_ignores_pins():
    """evict/evict_matching are intentional lifecycle events; pins don't apply."""

    @closing_lru_cache(maxsize=2, lease=False, pinned_predicate=lambda key: True)
    def create(name):
        return _Closeable(name)

    value_a = create("a")

    assert create.cache_evict("a") is True
    assert value_a.closed is True
    assert create.cache_info().currsize == 0


def test_overflow_converges_back_after_unpin():
    """An overflowed cache evicts down below maxsize once entries unpin."""
    pinned = {("a",), ("b",)}

    @closing_lru_cache(maxsize=2, lease=False, pinned_predicate=lambda key: key in pinned)
    def create(name):
        return _Closeable(name)

    value_a = create("a")
    create("b")
    value_c = create("c")  # both pinned -> overflow to 3
    assert create.cache_info().currsize == 3

    pinned.discard(("a",))
    create("d")  # evicts "a" (unpinned LRU), then "c" to get back under maxsize

    assert value_a.closed is True
    assert value_c.closed is True
    assert create.cache_info().currsize == 2
    assert create.cache_contains("b")
    assert create.cache_contains("d")


def test_aget_does_not_wait_for_deferred_close_behind_live_proxy():
    """A close deferred behind a live caller proxy is NOT waited on: idle
    holders can keep a proxy alive indefinitely (and may include the waiting
    caller itself), so a creator must proceed immediately. The deferred close
    still runs — and drains from the registry — once the holder lets go."""
    import asyncio

    cache = ClosingLRUCache(maxsize=8, lease=True)
    closeable = _Closeable("old")
    holder = {"proxy": cache.get_or_create("k", lambda: closeable)}
    assert cache.evict("k") is True  # close deferred: proxy still held

    async def scenario():
        new_value = await asyncio.wait_for(
            cache.aget_or_create("k", lambda: _Closeable("new")), timeout=1
        )
        assert new_value.name == "new"
        assert closeable.closed is False  # old value still open behind the holder

        holder.pop("proxy")
        gc.collect()  # run the proxy finalizer -> deferred close starts + resolves
        await cache.await_pending_closes()

    asyncio.run(scenario())
    assert closeable.closed is True
    assert not cache._closing


def test_sync_get_or_create_blocks_for_in_flight_close_off_loop():
    """Without a running loop, sync creation blocks until an IN-FLIGHT close
    of the previous value for the key has fully completed — but never waits
    for a close that is still deferred behind a live proxy."""
    import threading

    cache = ClosingLRUCache(maxsize=8, lease=True)
    started, release = threading.Event(), threading.Event()
    worker = _SlowSubprocessWorker("old", started=started, release=release)
    proxy = cache.get_or_create("k", lambda: worker)
    del proxy
    gc.collect()  # no holders: evict starts the close immediately (in flight)
    assert cache.evict("k") is True
    assert started.wait(5), "close did not start"

    def release_later():
        started.wait(5)
        release.set()

    releaser = threading.Thread(target=release_later)
    releaser.start()
    new_value = cache.get_or_create("k", lambda: _Closeable("new"))
    releaser.join()

    assert worker.closed is True, "creator returned before the in-flight close finished"
    assert new_value.name == "new"


def test_sync_get_or_create_does_not_block_for_deferred_close():
    """A deferred close (proxy still held) must not block a sync creator."""
    import time

    cache = ClosingLRUCache(maxsize=8, lease=True)
    closeable = _Closeable("old")
    holder = {"proxy": cache.get_or_create("k", lambda: closeable)}
    assert cache.evict("k") is True  # close deferred behind the held proxy

    start = time.monotonic()
    new_value = cache.get_or_create("k", lambda: _Closeable("new"))
    assert time.monotonic() - start < 1.0, "sync creator blocked on a deferred close"
    assert new_value.name == "new"
    assert closeable.closed is False

    holder.pop("proxy")
    gc.collect()
    assert closeable.closed is True


def test_cache_clear_ignores_pins():
    """cache_clear is an intentional lifecycle event: pinned entries close too."""

    @closing_lru_cache(maxsize=4, lease=False, pinned_predicate=lambda key: True)
    def create(name):
        return _Closeable(name)

    values = [create(name) for name in ("a", "b")]
    create.cache_clear()

    assert all(value.closed is True for value in values)
    assert create.cache_info().currsize == 0


def test_pinning_with_maxsize_one_converges():
    """Extreme bound: maxsize=1 with a pinned entry overflows, then converges
    to a single entry once the pin lifts."""
    pinned = {("a",)}

    @closing_lru_cache(maxsize=1, lease=False, pinned_predicate=lambda key: key in pinned)
    def create(name):
        return _Closeable(name)

    value_a = create("a")
    value_b = create("b")  # pinned "a" -> overflow to 2
    assert create.cache_info().currsize == 2

    pinned.clear()
    create("c")  # evicts "a" and "b" to get back under maxsize

    assert value_a.closed is True
    assert value_b.closed is True
    assert create.cache_info().currsize == 1
    assert create.cache_contains("c")


def test_name_bound_pin_predicate_resolves_parameter():
    """A DatasetQueuePinPredicate addresses the key by parameter NAME; the
    decorator binds it to the real signature, and pinning follows the queue."""
    from unittest.mock import MagicMock, patch

    from cognee.infrastructure.databases.dataset_queue.pinning import dataset_queue_pin_predicate

    predicate = dataset_queue_pin_predicate("db_name")

    @closing_lru_cache(maxsize=1, lease=False, pinned_predicate=predicate)
    def create(provider, db_name):
        return _Closeable(f"{provider}/{db_name}")

    fake_queue = MagicMock()
    fake_queue.active_dataset_ids.return_value = {"dataset-1"}
    with patch(
        "cognee.infrastructure.databases.dataset_queue.dataset_queue",
        return_value=fake_queue,
    ):
        pinned_value = create("kuzu", "dataset-1.pkl")  # active -> pinned
        create("kuzu", "other.pkl")  # would evict LRU, but it is pinned -> overflow

    assert pinned_value.closed is False
    assert create.cache_info().currsize == 2


def test_name_bound_pin_predicate_rejects_unknown_parameter():
    """A typo'd parameter name fails at decoration time, not silently at runtime."""
    import pytest

    from cognee.infrastructure.databases.dataset_queue.pinning import dataset_queue_pin_predicate

    with pytest.raises(ValueError, match="no_such_param"):

        @closing_lru_cache(maxsize=2, pinned_predicate=dataset_queue_pin_predicate("no_such_param"))
        def create(provider, db_name):
            return _Closeable(db_name)


class _LifecycleEngine:
    """Closeable stub asserting the close→open ordering invariant: a new
    engine for a key may only be constructed after its predecessor closed."""

    def __init__(self, key, registry):
        self.key = key
        self.closed = False
        predecessor = registry.get(key)
        assert predecessor is None or predecessor.closed, (
            f"new engine for {key} constructed while predecessor still open"
        )
        registry[key] = self

    async def close(self):
        import asyncio

        await asyncio.sleep(0)  # force the async (loop-task) close path
        self.closed = True


def test_hundred_keys_with_rotating_pins_never_close_active():
    """100 keys through a maxsize-6 cache with a rotating pinned window:
    pinned entries survive, the cache converges once pins lift, and every
    evicted engine is closed."""
    import asyncio

    registry: dict = {}
    pinned: set = set()
    created: list = []

    @closing_lru_cache(maxsize=6, lease=False, pinned_predicate=lambda key: key[0] in pinned)
    def create(name):
        engine = _LifecycleEngine(name, registry)
        created.append(engine)
        return engine

    async def scenario():
        for i in range(100):
            pinned.clear()
            pinned.update(f"k{j}" for j in range(max(0, i - 4), i + 1))
            create(f"k{i}")
            assert create.cache_info().currsize <= 6 + len(pinned)
            await asyncio.sleep(0)  # let scheduled async closes run
        pinned.clear()
        create("final")  # trigger convergence back under maxsize
        for _ in range(3):
            await asyncio.sleep(0)
        assert create.cache_info().currsize <= 6
        closed = sum(1 for engine in created if engine.closed)
        assert closed >= len(created) - 6 - 1

    asyncio.run(scenario())


def test_hundred_evict_recreate_cycles_hold_close_open_ordering():
    """100 randomized evict→recreate cycles over contended keys, half with
    the close deferred behind a held lease: the factory-level ordering
    assertion must hold every time and the pending-close registry must drain."""
    import asyncio
    import gc
    import random

    random.seed(42)
    registry: dict = {}
    cache = ClosingLRUCache(maxsize=4, lease=True)

    async def scenario():
        for i in range(100):
            key = f"key{i % 5}"
            holder = await cache.aget_or_create(key, lambda k=key: _LifecycleEngine(k, registry))
            if random.random() < 0.5:
                cache.evict(key)  # close deferred behind `holder`
                del holder
                gc.collect()
            else:
                del holder
                cache.evict(key)
            await asyncio.sleep(0)
        await cache.await_pending_closes()
        assert not cache._closing

    asyncio.run(scenario())


def test_hundred_concurrent_creators_close_everything_and_drain_registry():
    """100 concurrent creators across 10 contended keys with interleaved
    evictions. Strict close→open ordering is NOT guaranteed here — an eviction
    can land while another worker still holds the shared proxy, and creators
    deliberately do not wait for closes deferred behind live holders — but
    every superseded engine must eventually close and the pending-close
    registry must drain."""
    import asyncio
    import gc
    import random

    random.seed(42)
    created: list = []
    cache = ClosingLRUCache(maxsize=6, lease=True)

    def make(key):
        engine = _Closeable(key)
        created.append(engine)
        return engine

    async def worker(i):
        key = f"c{i % 10}"
        value = await cache.aget_or_create(key, lambda k=key: make(k))
        await asyncio.sleep(random.random() * 0.005)
        if i % 3 == 0:
            cache.evict(key)
        del value

    async def scenario():
        await asyncio.gather(*(worker(i) for i in range(100)))
        cache.cache_clear()
        gc.collect()
        await asyncio.sleep(0.05)
        await cache.await_pending_closes()
        assert not cache._closing
        assert all(engine.closed for engine in created)

    asyncio.run(scenario())


# -- Pending-close resilience: cancelled, stranded, or lost closes -----------


def test_close_cancelled_at_loop_exit_does_not_wedge_next_creation():
    """Regression for the CI hang shipped with the pending-close registry: an
    async close scheduled as a loop task is cancelled by ``asyncio.run`` at
    loop teardown if still pending. The done-callback must resolve the registry
    entry even on cancellation — otherwise the next creation for the same key
    waits on it forever."""
    import asyncio

    cache = ClosingLRUCache(maxsize=8, lease=True)

    class _NeverFinishesClose:
        async def close(self):
            await asyncio.sleep(3600)  # cancelled at loop teardown

    async def evict_and_exit():
        proxy = cache.get_or_create("k", _NeverFinishesClose)
        del proxy
        gc.collect()  # release the lease so evict() starts the close now
        assert cache.evict("k") is True
        # Exit while the close task is still pending: asyncio.run cancels it.

    asyncio.run(evict_and_exit())

    async def recreate():
        return await asyncio.wait_for(
            cache.aget_or_create("k", lambda: _Closeable("new")), timeout=5
        )

    new_value = asyncio.run(recreate())
    assert new_value.name == "new"


def test_wait_for_pending_close_is_bounded(monkeypatch):
    """A pending close that never resolves (wedged close, lost completion
    signal) delays a creator by at most ``PENDING_CLOSE_WAIT_SECONDS``, then
    creation proceeds with a warning instead of hanging."""
    import asyncio
    import concurrent.futures
    from cognee.infrastructure.databases.utils import closing_lru_cache as cache_module

    monkeypatch.setattr(cache_module, "PENDING_CLOSE_WAIT_SECONDS", 0.2)

    cache = ClosingLRUCache(maxsize=8, lease=True)
    lost = concurrent.futures.Future()  # a registered close that never resolves
    with cache._lock:
        cache._closing["k"] = lost

    async def recreate():
        return await asyncio.wait_for(
            cache.aget_or_create("k", lambda: _Closeable("new")), timeout=5
        )

    new_value = asyncio.run(recreate())
    assert new_value.name == "new"
    assert lost.done() is False  # we gave up waiting; the note itself is untouched


# -- Exception surfacing: unexpected failures must never be silent -----------

_CACHE_LOGGER = "cognee.infrastructure.databases.utils.closing_lru_cache"


def test_sync_close_failure_is_logged(caplog):
    """A sync close() that raises is logged with the traceback, and the
    eviction still succeeds."""
    import logging

    class _RaisingSyncClose:
        def close(self):
            raise RuntimeError("boom in sync close")

    cache = ClosingLRUCache(maxsize=4, lease=False)
    cache.get_or_create("k", _RaisingSyncClose)
    with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
        assert cache.evict("k") is True

    matching = [r for r in caplog.records if "Failed to close" in r.getMessage()]
    assert matching and matching[0].exc_info is not None


def test_subprocess_close_failure_is_logged(caplog):
    """An async close() raising on the dedicated close thread pool
    (subprocess-mode adapters) is logged with the traceback, and the pending
    registry still drains."""
    import asyncio
    import logging

    class _RaisingSubprocessClose:
        _subprocess_mode = True

        async def close(self):
            raise RuntimeError("boom in subprocess close")

    cache = ClosingLRUCache(maxsize=4, lease=False)
    cache.get_or_create("k", _RaisingSubprocessClose)

    async def scenario():
        cache.evict("k")
        await cache.await_pending_closes()

    with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
        asyncio.run(scenario())

    matching = [r for r in caplog.records if "Failed to run async close()" in r.getMessage()]
    assert matching and matching[0].exc_info is not None
    assert not cache._closing


def test_async_close_failure_without_loop_is_logged(caplog):
    """An async close() raising in the no-running-loop fallback branch
    (``asyncio.run``) is logged with the traceback."""
    import logging

    class _RaisingAsyncClose:
        async def close(self):
            raise RuntimeError("boom in async close, no loop")

    cache = ClosingLRUCache(maxsize=4, lease=False)
    cache.get_or_create("k", _RaisingAsyncClose)
    with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
        assert cache.evict("k") is True

    matching = [r for r in caplog.records if "Failed to run async close()" in r.getMessage()]
    assert matching and matching[0].exc_info is not None
    assert not cache._closing


def test_registry_future_carrying_exception_is_logged_and_creation_proceeds(caplog):
    """If the invariant 'close futures resolve with a result, never an
    exception' ever breaks, the waiter must log the exception with its
    traceback and still create the new value — never swallow, never fail."""
    import asyncio
    import concurrent.futures
    import logging

    cache = ClosingLRUCache(maxsize=8, lease=True)
    broken = concurrent.futures.Future()
    broken.set_exception(RuntimeError("invariant broke"))
    with cache._lock:
        cache._closing["k"] = broken

    # The future is done, so the fast path skips the wait; register a
    # not-yet-done broken future via a delayed set_exception instead.
    pending = concurrent.futures.Future()
    with cache._lock:
        cache._closing["k2"] = pending

    async def scenario():
        asyncio.get_running_loop().call_later(
            0.05, pending.set_exception, RuntimeError("late boom")
        )
        return await asyncio.wait_for(
            cache.aget_or_create("k2", lambda: _Closeable("new")), timeout=5
        )

    with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
        new_value = asyncio.run(scenario())

    assert new_value.name == "new"
    matching = [r for r in caplog.records if "Unexpected error while waiting" in r.getMessage()]
    assert matching and matching[0].exc_info is not None


def test_cancelled_registry_future_is_logged_and_creation_proceeds(caplog):
    """A cancelled registry future (broken invariant) must be surfaced and
    must not abort the creator masquerading as caller cancellation."""
    import asyncio
    import concurrent.futures
    import logging

    cache = ClosingLRUCache(maxsize=8, lease=True)
    pending = concurrent.futures.Future()
    with cache._lock:
        cache._closing["k"] = pending

    async def scenario():
        asyncio.get_running_loop().call_later(0.05, pending.cancel)
        return await asyncio.wait_for(
            cache.aget_or_create("k", lambda: _Closeable("new")), timeout=5
        )

    with caplog.at_level(logging.WARNING, logger=_CACHE_LOGGER):
        new_value = asyncio.run(scenario())

    assert new_value.name == "new"
    assert any("cancelled unexpectedly" in r.getMessage() for r in caplog.records)


def test_caller_cancellation_propagates_while_waiting_for_close():
    """Genuine cancellation of the waiting caller's task must propagate out of
    ``aget_or_create`` — it is the caller's exception, not ours to eat."""
    import asyncio
    import concurrent.futures

    cache = ClosingLRUCache(maxsize=8, lease=True)
    pending = concurrent.futures.Future()  # never resolves
    with cache._lock:
        cache._closing["k"] = pending

    async def scenario():
        task = asyncio.ensure_future(cache.aget_or_create("k", lambda: _Closeable("new")))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "not cancelled"

    assert asyncio.run(scenario()) == "cancelled"

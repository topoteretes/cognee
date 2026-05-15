"""Tests for ClosingLRUCache and the @closing_lru_cache decorator."""

import gc

from cognee.infrastructure.databases.utils.closing_lru_cache import (
    ClosingLRUCache,
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

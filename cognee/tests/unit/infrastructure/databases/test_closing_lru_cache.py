"""Tests for ClosingLRUCache and the @closing_lru_cache decorator."""

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
    """When maxsize is exceeded, the oldest entry is evicted and closed."""
    cache = ClosingLRUCache(maxsize=2)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    cache.get_or_create("b", lambda: _Closeable("b"))
    cache.get_or_create("c", lambda: _Closeable("c"))

    assert a.closed is True
    assert cache.cache_info()["size"] == 2


def test_access_refreshes_lru_order():
    """Accessing a key moves it to most-recently-used position."""
    cache = ClosingLRUCache(maxsize=2)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    b = cache.get_or_create("b", lambda: _Closeable("b"))

    # Access "a" to refresh it
    cache.get_or_create("a", lambda: _Closeable("should not create"))

    # "c" evicts "b" (now oldest), not "a"
    cache.get_or_create("c", lambda: _Closeable("c"))

    assert a.closed is False
    assert b.closed is True


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


# -- ClosingLRUCache: cache_clear --------------------------------------------


def test_cache_clear_closes_all_entries():
    """cache_clear calls close() on every cached value and empties the cache."""
    cache = ClosingLRUCache(maxsize=4)
    a = cache.get_or_create("a", lambda: _Closeable("a"))
    b = cache.get_or_create("b", lambda: _Closeable("b"))
    c = cache.get_or_create("c", lambda: _Closeable("c"))

    cache.cache_clear()

    assert a.closed is True
    assert b.closed is True
    assert c.closed is True
    assert cache.cache_info()["size"] == 0


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
    """Decorated function evicts oldest entry and calls close() on it."""
    @closing_lru_cache(maxsize=2)
    def create(key):
        return _Closeable(key)

    a = create("a")
    create("b")
    create("c")

    assert a.closed is True


def test_decorator_exposes_cache_clear():
    """Decorated function has a cache_clear method that closes all entries."""
    @closing_lru_cache(maxsize=4)
    def create(key):
        return _Closeable(key)

    a = create("a")
    create.cache_clear()
    assert a.closed is True


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

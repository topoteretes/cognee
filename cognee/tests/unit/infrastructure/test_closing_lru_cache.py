"""Tests for ``closing_lru_cache``.

Covers:
- LRU eviction calls sync / async ``close()`` exactly once
- ``cache_clear()`` closes all current entries and resets stats
- ``cache_evict()`` returns True/False and closes only when present
- Values without ``close`` are dropped silently
- Errors inside ``close()`` are logged but do not propagate, entry still removed
- Concurrent ``wrapper()`` calls are thread-safe; thundering-herd loser is closed
- Bare-decorator and parametrized-decorator forms both work
- ``typed=True`` distinguishes ``f(3)`` and ``f(3.0)``
- Async ``close`` runs on the caller's loop when one is present
"""

import asyncio
import threading
from unittest.mock import patch

import pytest

from cognee.infrastructure.utils.closing_lru_cache import closing_lru_cache


class Closeable:
    """Test resource that records sync close() calls."""

    instances: list["Closeable"] = []

    def __init__(self, label):
        self.label = label
        self.close_count = 0
        Closeable.instances.append(self)

    def close(self):
        self.close_count += 1


class AsyncCloseable:
    """Test resource that records async close() calls."""

    def __init__(self, label):
        self.label = label
        self.close_count = 0
        self.closed_on_loop_id = None

    async def close(self):
        self.close_count += 1
        self.closed_on_loop_id = id(asyncio.get_running_loop())


class NoCloseAttr:
    """Test resource without a close attribute."""

    def __init__(self, label):
        self.label = label


@pytest.fixture(autouse=True)
def _reset_closeable_instances():
    Closeable.instances = []
    yield
    Closeable.instances = []


# ---------------------------------------------------------------------------
# Basic decorator forms
# ---------------------------------------------------------------------------


def test_bare_decorator_form():
    @closing_lru_cache
    def make(x):
        return Closeable(x)

    a = make(1)
    b = make(1)
    assert a is b
    info = make.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["maxsize"] == 128


def test_parametrized_decorator_form():
    @closing_lru_cache(maxsize=2)
    def make(x):
        return Closeable(x)

    make(1)
    make(2)
    assert make.cache_info()["maxsize"] == 2


def test_typed_distinguishes_int_and_float():
    @closing_lru_cache(typed=True)
    def make(x):
        return Closeable(x)

    a = make(3)
    b = make(3.0)
    assert a is not b
    assert make.cache_info()["currsize"] == 2


def test_untyped_collapses_int_and_float():
    @closing_lru_cache(typed=False)
    def make(x):
        return Closeable(x)

    a = make(3)
    b = make(3.0)
    assert a is b
    assert make.cache_info()["currsize"] == 1


def test_kwargs_distinguished_from_positional():
    @closing_lru_cache
    def make(x):
        return Closeable(x)

    a = make(1)
    b = make(x=1)
    # Positional and keyword forms must NOT collide — _KWARGS_MARK separates them.
    assert a is not b
    assert make.cache_info()["currsize"] == 2


# ---------------------------------------------------------------------------
# Eviction behavior
# ---------------------------------------------------------------------------


def test_lru_eviction_calls_sync_close_once():
    @closing_lru_cache(maxsize=2)
    def make(x):
        return Closeable(x)

    a = make(1)
    make(2)
    make(3)  # evicts `a`

    assert a.close_count == 1
    assert make.cache_info()["currsize"] == 2


def test_lru_eviction_uses_lru_order_not_insertion_order():
    @closing_lru_cache(maxsize=2)
    def make(x):
        return Closeable(x)

    a = make(1)
    b = make(2)
    make(1)  # bumps `a` to most-recent
    make(3)  # should evict `b`, not `a`

    assert a.close_count == 0
    assert b.close_count == 1


def test_lru_eviction_calls_async_close_once():
    @closing_lru_cache(maxsize=1)
    def make(x):
        return AsyncCloseable(x)

    a = make(1)
    make(2)  # evicts `a`

    assert a.close_count == 1


def test_value_without_close_attr_is_dropped_silently():
    @closing_lru_cache(maxsize=1)
    def make(x):
        return NoCloseAttr(x)

    a = make(1)
    make(2)  # would close `a` if it had close — should just drop

    # No exception, no assertion on close_count (attribute doesn't exist)
    assert not hasattr(a, "close")


def test_close_exception_logged_but_not_propagated():
    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    @closing_lru_cache(maxsize=1)
    def make(x):
        return BadClose()

    make(1)
    # Should not raise even though close() raises
    make(2)


def test_close_exception_does_not_block_subsequent_evictions():
    closed_labels = []

    class MaybeBadClose:
        def __init__(self, label, should_raise):
            self.label = label
            self.should_raise = should_raise

        def close(self):
            if self.should_raise:
                raise RuntimeError(f"boom-{self.label}")
            closed_labels.append(self.label)

    @closing_lru_cache(maxsize=10)
    def make(label, should_raise):
        return MaybeBadClose(label, should_raise)

    make("a", True)
    make("b", False)
    make("c", False)

    make.cache_clear()

    # `a` raised but `b` and `c` should still have closed.
    assert sorted(closed_labels) == ["b", "c"]


# ---------------------------------------------------------------------------
# Explicit clear / evict
# ---------------------------------------------------------------------------


def test_cache_clear_closes_all_entries_and_resets_stats():
    @closing_lru_cache(maxsize=4)
    def make(x):
        return Closeable(x)

    a = make(1)
    b = make(2)
    make(1)  # one hit, to seed stats

    info_before = make.cache_info()
    assert info_before["hits"] == 1
    assert info_before["currsize"] == 2

    make.cache_clear()

    assert a.close_count == 1
    assert b.close_count == 1
    info_after = make.cache_info()
    assert info_after == {"hits": 0, "misses": 0, "maxsize": 4, "currsize": 0}


def test_cache_evict_returns_true_when_present_and_closes():
    @closing_lru_cache
    def make(x):
        return Closeable(x)

    a = make(1)
    assert make.cache_evict(1) is True
    assert a.close_count == 1
    assert make.cache_info()["currsize"] == 0


def test_cache_evict_returns_false_when_missing():
    @closing_lru_cache
    def make(x):
        return Closeable(x)

    make(1)
    assert make.cache_evict(99) is False
    assert make.cache_info()["currsize"] == 1


def test_cache_evict_with_kwargs():
    @closing_lru_cache
    def make(x, y=0):
        return Closeable((x, y))

    a = make(1, y=2)
    assert make.cache_evict(1, y=2) is True
    assert a.close_count == 1


# ---------------------------------------------------------------------------
# maxsize=None disables eviction
# ---------------------------------------------------------------------------


def test_unbounded_cache_does_not_evict():
    @closing_lru_cache(maxsize=None)
    def make(x):
        return Closeable(x)

    instances = [make(i) for i in range(50)]
    assert make.cache_info()["currsize"] == 50
    assert all(inst.close_count == 0 for inst in instances)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_thundering_herd_loser_value_is_closed():
    """When two threads compute the same key concurrently, the loser must be closed."""
    barrier = threading.Barrier(2)
    creation_lock = threading.Lock()
    created = []

    def factory(x):
        # Force both threads to start factory work simultaneously, then
        # serialize to make the race deterministic.
        barrier.wait()
        with creation_lock:
            instance = Closeable(x)
            created.append(instance)
            return instance

    @closing_lru_cache
    def make(x):
        return factory(x)

    results = [None, None]

    def worker(idx):
        results[idx] = make(1)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both threads got the same instance (the cache deduplicated).
    assert results[0] is results[1]

    # Two factory invocations occurred (the unavoidable lru_cache thundering-herd),
    # but exactly one of the produced instances must be closed.
    assert len(created) == 2
    closed = [inst for inst in created if inst.close_count == 1]
    not_closed = [inst for inst in created if inst.close_count == 0]
    assert len(closed) == 1
    assert len(not_closed) == 1
    # The surviving (uncllosed) instance is the one that callers received.
    assert results[0] is not_closed[0]


def test_concurrent_clear_and_get_does_not_crash():
    @closing_lru_cache(maxsize=8)
    def make(x):
        return Closeable(x)

    stop = threading.Event()
    errors = []

    def reader():
        try:
            while not stop.is_set():
                make(1)
                make(2)
        except Exception as e:
            errors.append(e)

    def clearer():
        try:
            for _ in range(50):
                make.cache_clear()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(2)] + [
        threading.Thread(target=clearer)
    ]
    for t in threads:
        t.start()
    threads[-1].join()
    stop.set()
    for t in threads[:-1]:
        t.join()

    assert errors == []


# ---------------------------------------------------------------------------
# Async close routing
# ---------------------------------------------------------------------------


def test_async_close_runs_on_caller_loop_when_one_exists():
    """When eviction happens inside a running loop, the async close must be
    scheduled on that same loop (not orphaned onto a fresh one), so the
    coroutine runs on the loop the resource is bound to.

    ``create_task`` is used (not blocking), so the close completes the
    next time the loop yields — we await a tick and then verify.
    """

    @closing_lru_cache(maxsize=1)
    def make(x):
        return AsyncCloseable(x)

    async def run():
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        a = make(1)
        make(2)  # evicts `a` — sync wrapper on the loop's thread

        # Allow the scheduled close task to run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        return a, loop_id

    a, loop_id = asyncio.run(run())
    assert a.close_count == 1
    assert a.closed_on_loop_id == loop_id


def test_async_close_falls_back_to_asyncio_run_when_no_loop():
    """When eviction happens entirely outside any event loop, async close
    should be driven by ``asyncio.run`` on a fresh loop and complete
    synchronously before the eviction returns."""

    @closing_lru_cache(maxsize=1)
    def make(x):
        return AsyncCloseable(x)

    a = make(1)
    make(2)  # evicts `a`, no running loop anywhere — should complete sync

    assert a.close_count == 1
    assert a.closed_on_loop_id is not None


# ---------------------------------------------------------------------------
# Logging on close failure
# ---------------------------------------------------------------------------


def test_close_failure_is_logged():
    class BadClose:
        def close(self):
            raise RuntimeError("boom")

    @closing_lru_cache(maxsize=1)
    def make(x):
        return BadClose()

    make(1)
    with patch("cognee.infrastructure.utils.closing_lru_cache.logger") as mock_logger:
        make(2)  # evicts the BadClose
        assert mock_logger.warning.called
        message = mock_logger.warning.call_args[0][0]
        assert "Error closing evicted cache value" in message

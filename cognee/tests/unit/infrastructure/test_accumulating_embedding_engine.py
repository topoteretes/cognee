import asyncio

import pytest

from cognee.infrastructure.databases.vector.embeddings.AccumulatingEmbeddingEngine import (
    AccumulatingEmbeddingEngine,
)


class FakeEngine:
    """Inner engine that records every embed_text call and returns deterministic vectors."""

    def __init__(self, batch_size: int = 4, dimensions: int = 3):
        self._batch_size = batch_size
        self._dimensions = dimensions
        self.calls: list[list[str]] = []
        self.tokenizer = "fake-tokenizer"

    async def embed_text(self, text) -> list[list[float]]:
        self.calls.append(list(text))
        return [[float(i)] * self._dimensions for i in range(len(text))]

    def get_vector_size(self) -> int:
        return self._dimensions

    def get_batch_size(self) -> int:
        return self._batch_size


@pytest.mark.asyncio
async def test_concurrent_calls_are_coalesced_into_single_inner_call():
    """Concurrent sub-batch-size calls coalesce into one inner embed_text; results split per caller."""
    inner = FakeEngine(batch_size=4)
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.05)

    r1_task = asyncio.create_task(eng.embed_text(["a", "b"]))
    r2_task = asyncio.create_task(eng.embed_text(["c"]))
    r1, r2 = await asyncio.gather(r1_task, r2_task)

    # One merged call: ["a","b","c"] flushed by the 50ms timer.
    assert len(inner.calls) == 1
    assert inner.calls[0] == ["a", "b", "c"]
    # Results split per caller, preserving order.
    assert len(r1) == 2
    assert len(r2) == 1


@pytest.mark.asyncio
async def test_full_batch_flushes_immediately_without_waiting_for_timeout():
    """Two concurrent sub-batch-size calls summing to batch_size should
    trigger the in-queue full-batch flush, not the timer-based flush.

    Sending a single ``[a,b,c,d]`` call would hit the oversized-bypass
    path instead, which is covered by ``test_oversized_request_bypasses_queue``.
    """
    inner = FakeEngine(batch_size=4)
    # Set a long timeout so the test fails if we accidentally waited on it.
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=5.0)

    started = asyncio.get_running_loop().time()
    r1, r2 = await asyncio.gather(
        eng.embed_text(["a", "b"]),
        eng.embed_text(["c", "d"]),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert len(r1) == 2 and len(r2) == 2
    # One merged inner call: queue fills to batch_size and flushes immediately.
    assert len(inner.calls) == 1
    assert sum(len(c) for c in inner.calls) == 4
    # Should be near-instant (well below the 5s timeout).
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_oversized_request_bypasses_queue():
    """Requests with size >= batch_size bypass the queue and dispatch directly to inner engine."""
    inner = FakeEngine(batch_size=4)
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.05)

    r = await eng.embed_text(["a", "b", "c", "d", "e"])

    assert len(r) == 5
    assert inner.calls == [["a", "b", "c", "d", "e"]]


@pytest.mark.asyncio
async def test_overflow_flushes_existing_queue_before_new_request():
    """When a new request would exceed capacity, existing queued items flush before the new request is queued."""
    inner = FakeEngine(batch_size=4)
    # Overflow flush is synchronous (happens before any timer is consulted),
    # so a short timeout is enough — and avoids waiting on the timer for the
    # second sub-batch.
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.05)

    # Pending = 3. Next request adds 2 -> total 5 > 4 -> existing queue flushes.
    first = asyncio.create_task(eng.embed_text(["a", "b", "c"]))
    # Let the first request register in the queue.
    await asyncio.sleep(0)
    second = asyncio.create_task(eng.embed_text(["d", "e"]))

    r1, r2 = await asyncio.gather(first, second)

    assert len(r1) == 3
    assert len(r2) == 2
    # Two separate batches because of the overflow flush.
    assert inner.calls == [["a", "b", "c"], ["d", "e"]]


@pytest.mark.asyncio
async def test_inner_engine_error_propagates_to_every_waiter():
    """Exceptions raised by inner.embed_text propagate to all waiting callers."""

    class FailingInner(FakeEngine):
        async def embed_text(self, text):
            raise RuntimeError("upstream failure")

    eng = AccumulatingEmbeddingEngine(FailingInner(batch_size=4), flush_timeout_seconds=0.05)

    r1 = asyncio.create_task(eng.embed_text(["a"]))
    r2 = asyncio.create_task(eng.embed_text(["b"]))
    results = await asyncio.gather(r1, r2, return_exceptions=True)

    assert all(isinstance(r, RuntimeError) and "upstream failure" in str(r) for r in results)


@pytest.mark.asyncio
async def test_oversized_bypass_validates_response_length():
    """The oversized-bypass fast path must still verify vector count matches inputs."""

    class TruncatingInner(FakeEngine):
        async def embed_text(self, text):
            # Drop the last vector regardless of size.
            return [[0.0] * self._dimensions for _ in range(len(text) - 1)]

    eng = AccumulatingEmbeddingEngine(TruncatingInner(batch_size=4), flush_timeout_seconds=0.05)

    with pytest.raises(RuntimeError):
        await eng.embed_text(["a", "b", "c", "d", "e"])  # 5 >= batch_size=4


@pytest.mark.asyncio
async def test_inner_vector_count_mismatch_fails_all_waiters():
    """If inner returns wrong vector count, all waiters receive a RuntimeError."""

    class WrongCountInner(FakeEngine):
        async def embed_text(self, text):
            # Return one fewer vector than requested.
            return [[0.0] * self._dimensions for _ in range(len(text) - 1)]

    eng = AccumulatingEmbeddingEngine(WrongCountInner(batch_size=4), flush_timeout_seconds=0.05)

    results = await asyncio.gather(
        eng.embed_text(["a"]),
        eng.embed_text(["b"]),
        return_exceptions=True,
    )
    assert all(isinstance(r, RuntimeError) for r in results)


@pytest.mark.asyncio
async def test_attribute_forwarding():
    """Attributes not defined on AccumulatingEmbeddingEngine are forwarded to inner engine."""
    inner = FakeEngine(batch_size=4)
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.05)

    assert eng.get_vector_size() == 3
    assert eng.get_batch_size() == 4
    # __getattr__ forwards to inner.
    assert eng.tokenizer == "fake-tokenizer"


@pytest.mark.asyncio
async def test_empty_input_returns_empty_without_inner_call():
    """Empty input returns empty list without calling inner.embed_text."""
    inner = FakeEngine(batch_size=4)
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.05)

    assert await eng.embed_text([]) == []
    assert inner.calls == []


@pytest.mark.asyncio
async def test_cancelled_waiters_skip_api_call():
    """If every waiter cancels before dispatch runs, the inner engine isn't called."""
    inner_started = asyncio.Event()
    inner_release = asyncio.Event()

    class GatedInner(FakeEngine):
        async def embed_text(self, text):
            inner_started.set()
            await inner_release.wait()
            return await super().embed_text(text)

    inner = GatedInner(batch_size=4)
    eng = AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0.02)

    # First batch: fire and cancel after timeout dispatches but before inner completes.
    t1 = asyncio.create_task(eng.embed_text(["a"]))
    t2 = asyncio.create_task(eng.embed_text(["b"]))
    await inner_started.wait()  # First batch is in flight.
    t1.cancel()
    t2.cancel()
    inner_release.set()
    await asyncio.gather(t1, t2, return_exceptions=True)
    # First batch went through (cancellation arrived during the inner call).

    # Now: enqueue more requests and cancel them BEFORE the timer fires.
    inner.calls.clear()
    t3 = asyncio.create_task(eng.embed_text(["c"]))
    t4 = asyncio.create_task(eng.embed_text(["d"]))
    await asyncio.sleep(0)  # let them register
    t3.cancel()
    t4.cancel()
    await asyncio.gather(t3, t4, return_exceptions=True)
    # Wait long enough for the timer to fire and the short-circuit to run.
    await asyncio.sleep(0.1)

    assert inner.calls == [], "inner.embed_text should not be called when all waiters cancel"


def test_rejects_inner_with_zero_batch_size():
    """Constructor raises ValueError when inner engine has batch_size=0."""

    class NoBatchSize(FakeEngine):
        def get_batch_size(self) -> int:
            return 0

    with pytest.raises(ValueError):
        AccumulatingEmbeddingEngine(NoBatchSize(), flush_timeout_seconds=0.05)


def test_rejects_non_positive_flush_timeout():
    """Constructor raises ValueError when flush_timeout_seconds is non-positive."""
    inner = FakeEngine(batch_size=4)
    with pytest.raises(ValueError):
        AccumulatingEmbeddingEngine(inner, flush_timeout_seconds=0)

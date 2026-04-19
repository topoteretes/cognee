"""Unit tests for the cognee.events pub/sub module."""

import pytest

from cognee.events import (
    CogneeEvent,
    INGEST_AFTER,
    INGEST_BEFORE,
    QUERY_AFTER,
    QUERY_BEFORE,
    clear_listeners,
    emit,
    subscribe,
)


@pytest.fixture(autouse=True)
def _reset_listeners():
    """Ensure each test starts and ends with no registered listeners."""
    clear_listeners()
    yield
    clear_listeners()


@pytest.mark.asyncio
async def test_emit_with_no_listeners_is_noop():
    # Should not raise even with zero subscribers.
    await emit(INGEST_BEFORE, dataset_name="main_dataset")


@pytest.mark.asyncio
async def test_sync_listener_is_invoked():
    received: list[CogneeEvent] = []

    def listener(event: CogneeEvent) -> None:
        received.append(event)

    subscribe(listener)
    await emit(INGEST_BEFORE, dataset_name="main_dataset")

    assert len(received) == 1
    assert received[0].event_type == INGEST_BEFORE
    assert received[0].payload == {"dataset_name": "main_dataset"}
    assert received[0].timestamp is not None


@pytest.mark.asyncio
async def test_async_listener_is_awaited():
    received: list[CogneeEvent] = []

    async def listener(event: CogneeEvent) -> None:
        received.append(event)

    subscribe(listener)
    await emit(QUERY_AFTER, query_text="hello", result_count=3)

    assert len(received) == 1
    assert received[0].event_type == QUERY_AFTER
    assert received[0].payload["result_count"] == 3


@pytest.mark.asyncio
async def test_unsubscribe_stops_future_events():
    received: list[CogneeEvent] = []

    def listener(event: CogneeEvent) -> None:
        received.append(event)

    unsubscribe = subscribe(listener)
    await emit(INGEST_AFTER)
    unsubscribe()
    await emit(INGEST_AFTER)

    assert len(received) == 1


@pytest.mark.asyncio
async def test_listener_exception_does_not_break_others():
    received: list[CogneeEvent] = []

    def bad_listener(event: CogneeEvent) -> None:
        raise RuntimeError("boom")

    def good_listener(event: CogneeEvent) -> None:
        received.append(event)

    subscribe(bad_listener)
    subscribe(good_listener)

    # Should not raise, and good_listener must still observe the event.
    await emit(QUERY_BEFORE, query_text="x")

    assert len(received) == 1
    assert received[0].event_type == QUERY_BEFORE


@pytest.mark.asyncio
async def test_multiple_listeners_each_receive_events():
    calls_a = 0
    calls_b = 0

    def listener_a(event: CogneeEvent) -> None:
        nonlocal calls_a
        calls_a += 1

    async def listener_b(event: CogneeEvent) -> None:
        nonlocal calls_b
        calls_b += 1

    subscribe(listener_a)
    subscribe(listener_b)
    await emit(INGEST_BEFORE)
    await emit(INGEST_AFTER)

    assert calls_a == 2
    assert calls_b == 2

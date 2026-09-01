"""Eval capture hook contract (SDK-529): zero cost when off, non-blocking when on.

Covers the hot-path contract (no serialization/await in emit), lazy
auto-registration from the environment, overflow accounting, the BATCH_SIZE
wake-up, loop-boundary and flusher-less draining, and flusher resilience to
bad payloads, failing sinks, and cancellation.
"""

import asyncio
import os
import threading
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock
from uuid import uuid4

import pytest

from cognee.base_config import get_base_config
from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RUN_MANIFEST,
    KIND_SUMMARY_GENERATED,
    StorageSink,
    get_capture_config,
    hook,
)
from cognee.shared.data_models import KnowledgeGraph

pytestmark = pytest.mark.usefixtures("capture_reset")


def _graph(names: list[str]) -> KnowledgeGraph:
    # Superset of the fields both KnowledgeGraph variants (default / gemini) need.
    return KnowledgeGraph.model_validate(
        {
            "summary": "s",
            "description": "d",
            "nodes": [
                {"id": name, "name": name, "type": "Person", "description": "x", "label": "L"}
                for name in names
            ],
            "edges": [],
        }
    )


class _Explosive:
    """A payload whose serialization must never happen on the hot path."""

    def model_dump(self, *args, **kwargs):
        raise AssertionError("model_dump must not be called by emit()")


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# 1 + 2: off by default; auto-registration from the environment
# ---------------------------------------------------------------------------


def test_off_by_default_costs_nothing(monkeypatch):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)

    assert capture.is_active() is False
    capture.emit(KIND_SUMMARY_GENERATED, _Explosive())
    capture.emit(KIND_SUMMARY_GENERATED, {"a": 1}, payload_kind="json")

    assert not hook._buffer
    assert not hook._flushers
    assert hook._dropped == 0
    # note()/bump() are no-ops without a scope.
    capture.note("k", "v")
    capture.bump("c")
    assert capture.current_scope() is None


def test_auto_registers_storage_sink_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path))
    get_capture_config.cache_clear()
    get_base_config.cache_clear()
    try:
        # No emit() has happened yet: is_active() itself is the initialization point.
        assert capture.is_active() is True
        assert isinstance(hook._sink, StorageSink)
        assert get_capture_config().cognee_capture_dir == os.path.join(str(tmp_path), "capture")
    finally:
        get_base_config.cache_clear()


def test_initialization_failure_leaves_capture_off(monkeypatch):
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_SAMPLE_RATE", "7")
    get_capture_config.cache_clear()
    fake_logger = MagicMock()
    monkeypatch.setattr(hook, "logger", fake_logger)

    assert capture.is_active() is False
    assert hook._initialized is True
    fake_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# 3: fake sink receives everything, serialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_delivers_all_payload_kinds(fake_capture_sink):
    run_id = uuid4()
    graph = _graph(["alice"])

    capture.emit(KIND_EXTRACTION_CHUNK_GRAPH, graph, run_id=run_id)
    capture.emit(
        KIND_EXTRACTION_FUZZY_MATCH,
        {"matches": [{"a": "b"}]},
        payload_kind="json",
        run_id=run_id,
        stage="fuzzy",
    )
    capture.emit(KIND_SUMMARY_GENERATED, "a summary", payload_kind="text", run_id=run_id)

    await capture.drain()

    records = fake_capture_sink.records
    assert len(records) == 3
    by_kind = {record["kind"]: record for record in records}
    assert by_kind[KIND_EXTRACTION_CHUNK_GRAPH]["payload"] == graph.model_dump(mode="json")
    assert isinstance(by_kind[KIND_EXTRACTION_CHUNK_GRAPH]["payload"], dict)
    assert by_kind[KIND_EXTRACTION_FUZZY_MATCH]["payload"] == {"matches": [{"a": "b"}]}
    assert by_kind[KIND_EXTRACTION_FUZZY_MATCH]["stage"] == "fuzzy"
    assert by_kind[KIND_SUMMARY_GENERATED]["payload"] == "a summary"
    assert all(record["run_id"] == str(run_id) for record in records)
    assert all(record["dataset_id"] is None for record in records)
    assert all(isinstance(record["ts"], float) for record in records)
    # Sink batches are grouped by (run_id, kind).
    assert all(len({r["kind"] for r in call}) == 1 for call in fake_capture_sink.calls)


# ---------------------------------------------------------------------------
# 4: overflow drops newest and is accounted for in the manifest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_drops_newest_and_reports_delta(fake_capture_sink):
    hook._configure(queue_size=3)

    with capture.run_scope(uuid4(), kind="pipeline"):
        for index in range(5):
            capture.emit(KIND_SUMMARY_GENERATED, f"s{index}", payload_kind="text")
        assert hook._dropped == 2
        assert len(hook._buffer) == 3
        await capture.drain()
    await capture.drain()

    records = fake_capture_sink.records
    delivered = [r["payload"] for r in records if r["kind"] == KIND_SUMMARY_GENERATED]
    assert delivered == ["s0", "s1", "s2"]
    [manifest] = [r for r in records if r["kind"] == KIND_RUN_MANIFEST]
    assert manifest["payload"]["dropped_events"] == 2


# ---------------------------------------------------------------------------
# 5: the BATCH_SIZE trigger wakes the flusher without a drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_size_wakes_flusher_without_drain(fake_capture_sink):
    hook._configure(batch_size=4, flush_interval_s=60.0)

    with capture.run_scope(uuid4(), kind="pipeline"):
        for index in range(4):
            capture.emit(KIND_SUMMARY_GENERATED, f"s{index}", payload_kind="text")
        # No drain: the wake fired via call_soon_threadsafe, not the 60 s interval.
        await _wait_until(lambda: len(fake_capture_sink.records) >= 4)

    assert sorted(r["payload"] for r in fake_capture_sink.records) == ["s0", "s1", "s2", "s3"]
    [flusher] = hook._flushers.values()
    assert flusher.loop is asyncio.get_running_loop()
    assert not flusher.task.done()


# ---------------------------------------------------------------------------
# 9 + 10: loop boundaries and flusher-less draining
# ---------------------------------------------------------------------------


def test_emit_and_drain_across_asyncio_run_boundaries(fake_capture_sink):
    async def _round(tag):
        capture.emit(KIND_SUMMARY_GENERATED, tag, payload_kind="text")
        await capture.drain()

    asyncio.run(_round("first"))

    # A loop closed WITHOUT cancelling its flusher (pytest-asyncio 0.21 style)
    # leaves a stale entry; the next emit must prune it.
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    hook._flushers[stale_loop] = hook._Flusher(
        task=SimpleNamespace(done=lambda: False), loop=stale_loop, wake=asyncio.Event()
    )

    async def _second():
        capture.emit(KIND_SUMMARY_GENERATED, "second", payload_kind="text")
        await capture.drain()
        assert len(hook._flushers) == 1
        [flusher] = hook._flushers.values()
        assert flusher.loop is asyncio.get_running_loop()
        assert not flusher.loop.is_closed()

    asyncio.run(_second())

    assert [r["payload"] for r in fake_capture_sink.records] == ["first", "second"]


def test_drain_without_a_flusher_delivers_everything(fake_capture_sink):
    # No running loop here: emit buffers, starts nothing.
    capture.emit(KIND_SUMMARY_GENERATED, "sync", payload_kind="text")

    worker = threading.Thread(
        target=lambda: capture.emit(KIND_SUMMARY_GENERATED, "thread", payload_kind="text")
    )
    worker.start()
    worker.join()

    assert len(hook._buffer) == 2
    assert not hook._flushers

    asyncio.run(capture.drain())

    assert sorted(r["payload"] for r in fake_capture_sink.records) == ["sync", "thread"]
    assert not hook._buffer


# ---------------------------------------------------------------------------
# 11: the hot path never serializes; snapshots protect against later mutation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_never_serializes_on_the_hot_path(fake_capture_sink):
    capture.emit(KIND_EXTRACTION_CHUNK_GRAPH, _Explosive())
    assert len(hook._buffer) == 1
    # Held by reference until the flusher serializes it (off the hot path).
    assert isinstance(hook._buffer[0].payload, _Explosive)


@pytest.mark.asyncio
async def test_model_dump_snapshot_survives_later_mutation(fake_capture_sink):
    graph = _graph(["alice", "bob"])

    capture.emit(KIND_EXTRACTION_CHUNK_GRAPH, graph.model_dump(mode="json"), payload_kind="json")
    for node in graph.nodes:
        node.name = node.name.upper()

    await capture.drain()

    [record] = fake_capture_sink.records
    assert [node["name"] for node in record["payload"]["nodes"]] == ["alice", "bob"]


# ---------------------------------------------------------------------------
# 12: flusher resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_payload_becomes_error_record_and_flusher_survives(fake_capture_sink):
    hook._configure(batch_size=2, flush_interval_s=60.0)

    capture.emit(KIND_EXTRACTION_CHUNK_GRAPH, _Explosive())
    capture.emit(KIND_SUMMARY_GENERATED, "ok", payload_kind="text")
    await _wait_until(lambda: len(fake_capture_sink.records) >= 2)

    by_kind = {r["kind"]: r for r in fake_capture_sink.records}
    assert "AssertionError" in by_kind[KIND_EXTRACTION_CHUNK_GRAPH]["payload"]["error"]
    assert by_kind[KIND_SUMMARY_GENERATED]["payload"] == "ok"

    # The flusher task is still alive and keeps delivering.
    [flusher] = hook._flushers.values()
    assert not flusher.task.done()
    capture.emit(KIND_SUMMARY_GENERATED, "again", payload_kind="text")
    await capture.drain()
    assert "again" in [r["payload"] for r in fake_capture_sink.records]


@pytest.mark.asyncio
async def test_failing_sink_is_logged_at_debug_and_next_batch_delivered(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(hook, "logger", fake_logger)

    calls: list[list[dict]] = []

    async def flaky_sink(records):
        calls.append(records)
        if len(calls) == 1:
            raise RuntimeError("boom")

    capture.register_capture_sink(flaky_sink)

    capture.emit(KIND_SUMMARY_GENERATED, "one", payload_kind="text")
    await capture.drain()
    capture.emit(KIND_SUMMARY_GENERATED, "two", payload_kind="text")
    await capture.drain()

    assert [call[0]["payload"] for call in calls] == ["one", "two"]
    fake_logger.debug.assert_any_call("capture sink failed (%s)", ANY)
    assert hook._in_flight == 0


# ---------------------------------------------------------------------------
# 13: cancellation puts the popped batch back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_flush_requeues_popped_events(monkeypatch, fake_capture_sink):
    hook._configure(batch_size=1, flush_interval_s=60.0)
    entered = threading.Event()
    release = threading.Event()
    real_serialize = hook._serialize_batch

    def blocking_serialize(batch):
        entered.set()
        release.wait(timeout=5.0)
        return real_serialize(batch)

    monkeypatch.setattr(hook, "_serialize_batch", blocking_serialize)
    try:
        capture.emit(KIND_SUMMARY_GENERATED, "one", payload_kind="text")
        flusher = hook._flushers[asyncio.get_running_loop()]
        await _wait_until(entered.is_set)
        assert not hook._buffer  # popped, mid-serialization in the worker thread

        flusher.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flusher.task

        assert [event.payload for event in hook._buffer] == ["one"]
        assert asyncio.get_running_loop() not in hook._flushers
    finally:
        release.set()

    # A later drain still delivers the re-queued event.
    monkeypatch.setattr(hook, "_serialize_batch", real_serialize)
    await capture.drain()
    assert [r["payload"] for r in fake_capture_sink.records] == ["one"]


@pytest.mark.asyncio
async def test_shutdown_drains_and_stops_the_flusher(fake_capture_sink):
    capture.emit(KIND_SUMMARY_GENERATED, "last", payload_kind="text")
    [flusher] = hook._flushers.values()

    await capture.shutdown()

    assert [r["payload"] for r in fake_capture_sink.records] == ["last"]
    assert flusher.task.done()
    assert not hook._flushers

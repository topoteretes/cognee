"""Eval capture hook contract (SDK-529): zero cost when off, non-blocking when on.

Covers the hot-path contract (no serialization/await in emit), lazy
auto-registration from the environment, overflow accounting, the BATCH_SIZE
wake-up, loop-boundary and flusher-less draining, and flusher resilience to
bad payloads, failing sinks, and cancellation.
"""

import asyncio
import gzip
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
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
from cognee.modules.observability.capture.sinks import run_off_loop
from cognee.shared.data_models import KnowledgeGraph

pytestmark = pytest.mark.usefixtures("capture_reset")

REPO_ROOT = Path(__file__).resolve().parents[5]


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


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


@pytest.fixture
def block_first_serialize(monkeypatch):
    """Make the FIRST ``_serialize_batch`` call park in its worker thread.

    Returns ``(entered, release)`` threading.Events: ``entered`` fires when the
    flusher's batch is popped and stuck mid-serialization — the exact window
    where ``drain()``/``shutdown()`` used to see an empty buffer and return.
    """
    entered = threading.Event()
    release = threading.Event()
    real_serialize = hook._serialize_batch
    first = [True]

    def slow_first(batch):
        if first[0]:
            first[0] = False
            entered.set()
            release.wait(timeout=5.0)
        return real_serialize(batch)

    monkeypatch.setattr(hook, "_serialize_batch", slow_first)
    yield entered, release
    release.set()


# ---------------------------------------------------------------------------
# 1 + 2: off by default; auto-registration from the environment
# ---------------------------------------------------------------------------


# Async on purpose: with a running loop, "no flusher task created" is a claim
# that can fail (a sync test has no loop for _enqueue to start one on).
@pytest.mark.asyncio
@pytest.mark.parametrize("env_value", [None, "false"])
async def test_off_by_default_costs_nothing(monkeypatch, env_value):
    if env_value is None:
        monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", env_value)

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


@pytest.mark.asyncio
async def test_emit_before_any_is_active_never_initializes(monkeypatch):
    # With this set, an emit() that lazily initialized would auto-register a
    # sink and buffer the event; the contract is that only is_active() may.
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")

    def forbidden():
        raise AssertionError("emit() must never reach _ensure_initialized()")

    monkeypatch.setattr(hook, "_ensure_initialized", forbidden)

    capture.emit(KIND_SUMMARY_GENERATED, _Explosive())
    capture.emit(KIND_RUN_MANIFEST, {"kind": "pipeline"}, payload_kind="json")

    assert hook._initialized is False
    assert hook._sink is None
    assert not hook._buffer
    assert not hook._flushers


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


def test_register_capture_sink_keeps_the_caller_sink_but_loads_env_knobs(monkeypatch, tmp_path):
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("DATA_ROOT_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("COGNEE_CAPTURE_QUEUE_SIZE", "7")
    monkeypatch.setenv("COGNEE_CAPTURE_BATCH_SIZE", "3")
    monkeypatch.setenv("COGNEE_CAPTURE_FLUSH_INTERVAL_S", "0.25")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK_TIMEOUT_S", "3.5")
    get_capture_config.cache_clear()
    get_base_config.cache_clear()

    async def sink(records):
        pass

    try:
        # Registered BEFORE any is_active(): the explicit sink wins over env
        # auto-registration, but the tuning knobs still come from the environment.
        capture.register_capture_sink(sink)
    finally:
        get_base_config.cache_clear()

    assert hook._sink is sink
    assert hook.QUEUE_SIZE == 7
    assert hook.BATCH_SIZE == 3
    assert hook.FLUSH_INTERVAL_S == 0.25
    assert hook.SINK_TIMEOUT_S == 3.5


# ---------------------------------------------------------------------------
# 3: fake sink receives everything, serialized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_delivers_all_payload_kinds(fake_capture_sink):
    run_id = uuid4()
    other_run = uuid4()
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
    # Same kind under another run: must land in its own sink call.
    capture.emit(KIND_SUMMARY_GENERATED, "other run", payload_kind="text", run_id=other_run)

    await capture.drain()

    records = fake_capture_sink.records
    assert len(records) == 4
    by_kind = {record["kind"]: record for record in records if record["run_id"] == str(run_id)}
    assert by_kind[KIND_EXTRACTION_CHUNK_GRAPH]["payload"] == graph.model_dump(mode="json")
    assert isinstance(by_kind[KIND_EXTRACTION_CHUNK_GRAPH]["payload"], dict)
    assert by_kind[KIND_EXTRACTION_FUZZY_MATCH]["payload"] == {"matches": [{"a": "b"}]}
    assert by_kind[KIND_EXTRACTION_FUZZY_MATCH]["stage"] == "fuzzy"
    assert by_kind[KIND_SUMMARY_GENERATED]["payload"] == "a summary"
    [other] = [record for record in records if record["run_id"] == str(other_run)]
    assert other["payload"] == "other run"
    assert all(record["dataset_id"] is None for record in records)
    assert all(isinstance(record["ts"], float) for record in records)
    # Sink batches are grouped by (run_id, kind): four groups, one call each.
    assert len(fake_capture_sink.calls) == 4
    assert all(
        len({(r["run_id"], r["kind"]) for r in call}) == 1 for call in fake_capture_sink.calls
    )


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


@pytest.mark.asyncio
async def test_batch_wake_is_scheduled_once_per_synchronous_burst(monkeypatch, fake_capture_sink):
    hook._configure(batch_size=4, flush_interval_s=60.0)
    loop = asyncio.get_running_loop()
    real_call_soon_threadsafe = loop.call_soon_threadsafe
    scheduled = []

    def counting(callback, *args):
        scheduled.append(callback)
        return real_call_soon_threadsafe(callback, *args)

    monkeypatch.setattr(loop, "call_soon_threadsafe", counting)

    # A per-chunk loop that never yields: the scheduled wake.set() has not run
    # yet, so without the pending flag every emit past BATCH_SIZE re-schedules.
    for index in range(40):
        capture.emit(KIND_SUMMARY_GENERATED, f"s{index}", payload_kind="text")

    assert len(scheduled) == 1
    await _wait_until(lambda: len(fake_capture_sink.records) >= 40)
    [flusher] = hook._flushers.values()
    assert flusher.wake_pending is False


@pytest.mark.asyncio
async def test_scope_entry_starts_a_flusher_for_worker_thread_emits(fake_capture_sink):
    hook._configure(flush_interval_s=0.05)

    with capture.run_scope(uuid4(), kind="pipeline"):
        # Started eagerly at scope entry — before any on-loop emit.
        [flusher] = hook._flushers.values()
        assert flusher.loop is asyncio.get_running_loop()

        def worker():
            for index in range(5):
                capture.emit(KIND_SUMMARY_GENERATED, f"t{index}", payload_kind="text")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert len(hook._buffer) == 5
        # No drain, no on-loop emit: the interval tick alone delivers them.
        await _wait_until(lambda: len(fake_capture_sink.records) >= 5)

    assert sorted(r["payload"] for r in fake_capture_sink.records) == [f"t{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# 9 + 10: loop boundaries and flusher-less draining
# ---------------------------------------------------------------------------


def test_emit_and_drain_across_asyncio_run_boundaries(fake_capture_sink):
    async def _round(tag):
        capture.emit(KIND_SUMMARY_GENERATED, tag, payload_kind="text")
        await capture.drain()

    asyncio.run(_round("first"))
    # asyncio.run cancelled the flusher; the entry stays as a tombstone for the
    # closed loop until the next flusher start prunes it.
    [tombstone] = hook._flushers.values()
    assert tombstone.loop.is_closed()
    assert tombstone.task.cancelled()

    # A loop closed WITHOUT cancelling its flusher (pytest-asyncio 0.21 style)
    # leaves a stale entry; the next emit must prune it.
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    stale = hook._Flusher(loop=stale_loop, wake=asyncio.Event())
    stale.task = SimpleNamespace(done=lambda: False)
    hook._flushers[stale_loop] = stale

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


def test_emit_during_loop_teardown_starts_no_flusher_on_the_closing_loop(fake_capture_sink):
    hook._configure(batch_size=2, flush_interval_s=60.0)
    kept_alive: list = []
    seen: dict = {}

    async def pipeline_like():
        # run_tasks-shaped: the scope encloses the yields, so a generator its
        # consumer abandoned is finalized by asyncio.run's shutdown_asyncgens()
        # — AFTER _cancel_all_tasks() cancelled this loop's flusher.
        with capture.run_scope("run-1", "ds-1", kind="pipeline"):
            yield "started"
            yield "never reached"

    async def main():
        agen = pipeline_like()
        kept_alive.append(agen)  # not garbage-collected: finalized at loop teardown
        await agen.__anext__()
        # BATCH_SIZE - 1 buffered: the manifest emitted at teardown completes a batch.
        capture.emit(KIND_SUMMARY_GENERATED, "x", payload_kind="text", run_id="run-0")
        seen["flusher"] = hook._flushers[asyncio.get_running_loop()]

    asyncio.run(main())

    # The manifest landed while the loop was closing. A fresh flusher started
    # there would have popped the batch and been destroyed with the loop; the
    # cancelled flusher's tombstone keeps the events in the deque instead.
    flusher = seen["flusher"]
    assert flusher.task.cancelled() and flusher.loop.is_closed()
    assert hook._flushers[flusher.loop] is flusher
    assert [event.kind for event in hook._buffer] == [KIND_SUMMARY_GENERATED, KIND_RUN_MANIFEST]
    assert hook._in_flight_total() == 0
    assert hook._dropped == 0

    asyncio.run(capture.drain())

    kinds = [record["kind"] for record in fake_capture_sink.records]
    assert sorted(kinds) == sorted([KIND_SUMMARY_GENERATED, KIND_RUN_MANIFEST])
    [manifest] = [r for r in fake_capture_sink.records if r["kind"] == KIND_RUN_MANIFEST]
    assert manifest["run_id"] == "run-1" and manifest["dataset_id"] == "ds-1"
    assert hook._dropped == 0
    assert not hook._flushers  # the closed loop's tombstone was pruned by the drain


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

    capture.emit(
        KIND_EXTRACTION_CHUNK_GRAPH,
        graph.model_dump(mode="json"),
        payload_kind="json",
        stage="snapshot",
    )
    # Negative control: by reference, emit() holds the object and never copies
    # (a "helpful" deep copy inside emit() is forbidden on cost grounds).
    capture.emit(KIND_EXTRACTION_CHUNK_GRAPH, graph, payload_kind="pydantic", stage="reference")
    for node in graph.nodes:
        node.name = node.name.upper()

    await capture.drain()

    by_stage = {record["stage"]: record for record in fake_capture_sink.records}
    assert [n["name"] for n in by_stage["snapshot"]["payload"]["nodes"]] == ["alice", "bob"]
    assert [n["name"] for n in by_stage["reference"]["payload"]["nodes"]] == ["ALICE", "BOB"]


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
    fake_logger.debug.assert_any_call("capture sink failed, %d event(s) dropped (%s)", 1, ANY)
    assert hook._in_flight_total() == 0
    # The event the sink rejected is gone: it must be accounted for, not hidden,
    # or the run manifest reports dropped_events = 0 while records were lost.
    assert hook._dropped == 1
    assert not hook._buffer


@pytest.mark.asyncio
async def test_base_exception_from_a_sink_requeues_the_batch_and_the_flusher_is_replaced(
    monkeypatch,
):
    fake_logger = MagicMock()
    monkeypatch.setattr(hook, "logger", fake_logger)

    class Escaping(BaseException):
        pass

    calls: list[list[dict]] = []

    async def sink(records):
        calls.append(records)
        if len(calls) == 1:
            raise Escaping("boom")

    capture.register_capture_sink(sink)
    hook._configure(batch_size=2, flush_interval_s=60.0)

    capture.emit(KIND_SUMMARY_GENERATED, "a", payload_kind="text")
    capture.emit(KIND_SUMMARY_GENERATED, "b", payload_kind="text")  # BATCH_SIZE wake
    [flusher] = hook._flushers.values()
    await _wait_until(flusher.task.done)

    # Out of contract (sinks raise Exception subclasses only), but the batch is
    # not lost and the crash is retrieved — no "Task exception was never
    # retrieved" ERROR. A crashed (not cancelled) flusher is not a tombstone.
    assert not flusher.task.cancelled()
    assert [event.payload for event in hook._buffer] == ["a", "b"]
    assert hook._dropped == 0
    assert hook._in_flight_total() == 0
    assert asyncio.get_running_loop() not in hook._flushers
    fake_logger.debug.assert_any_call("capture flusher stopped (%r)", ANY)

    capture.emit(KIND_SUMMARY_GENERATED, "c", payload_kind="text")
    [replacement] = hook._flushers.values()
    assert replacement is not flusher and not replacement.task.done()
    await capture.drain()
    redelivered = [record["payload"] for call in calls[1:] for record in call]
    assert sorted(redelivered) == ["a", "b", "c"]


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
        # The cancelled entry stays as a tombstone: a cancelled flusher means
        # its loop is going away, and no replacement is started on it.
        assert hook._flushers[asyncio.get_running_loop()] is flusher
        assert hook._any_live_flusher() is None
    finally:
        release.set()

    # A later drain still delivers the re-queued event: drain needs no flusher.
    monkeypatch.setattr(hook, "_serialize_batch", real_serialize)
    await capture.drain()
    assert [r["payload"] for r in fake_capture_sink.records] == ["one"]


@pytest.mark.asyncio
async def test_shutdown_drains_and_stops_the_flusher(fake_capture_sink):
    capture.emit(KIND_SUMMARY_GENERATED, "last", payload_kind="text")
    [flusher] = hook._flushers.values()

    await capture.shutdown()

    assert [r["payload"] for r in fake_capture_sink.records] == ["last"]
    assert flusher.task.cancelled()
    loop = asyncio.get_running_loop()
    assert hook._flushers[loop] is flusher  # tombstone

    # An emit after shutdown (one more request during a lifespan shutdown) does
    # not resurrect a flusher; it stays buffered for the atexit hook.
    capture.emit(KIND_SUMMARY_GENERATED, "post", payload_kind="text")
    assert hook._flushers[loop] is flusher
    assert [event.payload for event in hook._buffer] == ["post"]

    # Registering a sink re-arms the loop: the next emit gets a fresh flusher.
    capture.register_capture_sink(fake_capture_sink)
    capture.emit(KIND_SUMMARY_GENERATED, "rearmed", payload_kind="text")
    assert hook._flushers[loop] is not flusher
    assert not hook._flushers[loop].task.done()
    await capture.drain()
    assert [r["payload"] for r in fake_capture_sink.records] == ["last", "post", "rearmed"]


# ---------------------------------------------------------------------------
# Cancellation is never swallowed (asyncio.wait_for on CPython < 3.12 loses a
# cancel that races the inner completion: a flusher that survives its cancel
# hangs asyncio.run() teardown and shutdown() forever)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_wait_propagates_a_cancel_racing_the_inner_completion():
    inner = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(hook._wait_bounded(inner, 60.0))
    await asyncio.sleep(0)  # parked on the timed wait

    # Completion and cancel land in the SAME loop iteration: here wait_for
    # returns the inner result and the cancel request is lost (bpo-42130).
    inner.set_result("done")
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert waiter.cancelled()


@pytest.mark.asyncio
async def test_bounded_wait_times_out_and_cancels_the_inner():
    inner_cancelled = asyncio.Event()

    async def inner():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            inner_cancelled.set()
            raise

    with pytest.raises(asyncio.TimeoutError):
        await hook._wait_bounded(inner(), 0.02)
    assert inner_cancelled.is_set()  # not left running after the timeout

    assert await hook._wait_bounded(asyncio.sleep(0, result="ok"), 1.0) == "ok"
    assert await hook._wait_bounded(asyncio.sleep(0, result="ok"), None) == "ok"


@pytest.mark.asyncio
async def test_flusher_cancelled_as_its_wake_lands_finishes_cancelled(fake_capture_sink):
    hook._configure(batch_size=4, flush_interval_s=60.0)
    hook.ensure_flusher()
    [flusher] = hook._flushers.values()
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # parked on its interval wait

    # The BATCH_SIZE wake arrives through the loop (as emit() sends it from any
    # thread) and the cancel lands right behind it — asyncio.run's teardown
    # timing when the last batch fills up as the command returns.
    asyncio.get_running_loop().call_soon_threadsafe(flusher.wake.set)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    flusher.task.cancel()

    done, _pending = await asyncio.wait({flusher.task}, timeout=1.0)
    assert flusher.task in done, "the flusher swallowed its cancellation and is still running"
    assert flusher.task.cancelled()


@pytest.mark.asyncio
async def test_drain_propagates_the_callers_cancellation():
    entered = asyncio.Event()
    release = asyncio.Event()
    delivered: list[dict] = []

    async def gated_sink(records):
        entered.set()
        await release.wait()
        delivered.extend(records)

    capture.register_capture_sink(gated_sink)
    hook._configure(flush_interval_s=60.0)
    for index in range(3):
        capture.emit(KIND_SUMMARY_GENERATED, f"c{index}", payload_kind="text")

    caller = asyncio.create_task(capture.drain(5.0))
    await entered.wait()  # drain popped the batch and is inside the sink write

    # The write completes and the cancel lands right behind it. A cancelled
    # pipeline run must not continue past its own cancellation (run_tasks
    # awaits drain() right before yielding PipelineRunCompleted).
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await caller
    assert caller.cancelled()
    assert [r["payload"] for r in delivered] == ["c0", "c1", "c2"]
    # The acknowledgement never ran, so the group is re-buffered: at-least-once.
    assert [event.payload for event in hook._buffer] == ["c0", "c1", "c2"]
    assert hook._in_flight_total() == 0


# ---------------------------------------------------------------------------
# drain()/shutdown() cover a batch the flusher is still serializing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_waits_for_a_batch_the_flusher_is_serializing(
    block_first_serialize, fake_capture_sink
):
    entered, release = block_first_serialize
    hook._configure(batch_size=4, flush_interval_s=60.0)

    for index in range(8):
        capture.emit(KIND_SUMMARY_GENERATED, f"e{index}", payload_kind="text")
    await _wait_until(entered.is_set)  # flusher popped 4, parked in the worker thread
    assert len(hook._buffer) == 4
    assert hook._in_flight_total() == 4  # counted per event, from pop to delivery

    async def release_later():
        await asyncio.sleep(0.1)
        release.set()

    releaser = asyncio.create_task(release_later())
    await capture.drain()
    await releaser

    # Both halves landed before drain returned: its own 4 inline, the flusher's 4 awaited.
    assert sorted(r["payload"] for r in fake_capture_sink.records) == [f"e{i}" for i in range(8)]
    assert hook._in_flight_total() == 0


@pytest.mark.asyncio
async def test_shutdown_recovers_a_batch_cancelled_mid_serialization(
    block_first_serialize, fake_capture_sink
):
    entered, _release = block_first_serialize
    hook._configure(batch_size=4, flush_interval_s=60.0)

    for index in range(8):
        capture.emit(KIND_SUMMARY_GENERATED, f"e{index}", payload_kind="text")
    await _wait_until(entered.is_set)

    # The flusher never gets released: drain times out, the cancel re-buffers
    # its batch, and shutdown must deliver that batch itself — no flusher is
    # left to do it and the atexit hook is a last resort, not the plan.
    await capture.shutdown(timeout=0.2)

    assert sorted(r["payload"] for r in fake_capture_sink.records) == [f"e{i}" for i in range(8)]
    assert not hook._buffer
    assert hook._any_live_flusher() is None
    assert hook._in_flight_total() == 0


@pytest.mark.asyncio
async def test_drain_timeout_bounds_it_under_a_continuous_producer():
    async def slow_sink(records):
        await asyncio.sleep(0.05)

    capture.register_capture_sink(slow_sink)
    hook._configure(batch_size=8, flush_interval_s=60.0, queue_size=10_000)
    stop = False

    async def producer():
        while not stop:
            for _ in range(20):
                capture.emit(KIND_SUMMARY_GENERATED, "x", payload_kind="text")
            await asyncio.sleep(0.01)

    producer_task = asyncio.create_task(producer())
    await asyncio.sleep(0.05)
    started = time.monotonic()
    await capture.drain(timeout=0.3)
    elapsed = time.monotonic() - started
    stop = True
    await producer_task

    # Producer outruns the sink, so an unbounded inline loop would never end.
    assert elapsed < 1.0, elapsed


@pytest.mark.asyncio
async def test_hung_sink_write_times_out_and_the_flusher_recovers(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(hook, "logger", fake_logger)
    calls: list[list[dict]] = []

    async def hanging_first(records):
        calls.append(records)
        if len(calls) == 1:
            await asyncio.Event().wait()  # never completes

    capture.register_capture_sink(hanging_first)
    hook._configure(batch_size=2, flush_interval_s=60.0, sink_timeout_s=0.1)

    capture.emit(KIND_SUMMARY_GENERATED, "e0", payload_kind="text")
    capture.emit(KIND_SUMMARY_GENERATED, "e1", payload_kind="text")
    await _wait_until(lambda: hook._in_flight_total() == 2)
    # The sink timeout, not a drain, frees the flusher.
    await _wait_until(lambda: hook._in_flight_total() == 0)
    fake_logger.debug.assert_any_call("capture sink timed out, %d event(s) dropped (%s)", 2, ANY)
    # The abandoned write's events are counted, not silently lost.
    assert hook._dropped == 2

    capture.emit(KIND_SUMMARY_GENERATED, "e2", payload_kind="text")
    started = time.monotonic()
    await capture.drain(timeout=2.0)
    assert time.monotonic() - started < 1.0  # not pinned by the abandoned write
    assert [r["payload"] for r in calls[1]] == ["e2"]
    [flusher] = hook._flushers.values()
    assert not flusher.task.done()


@pytest.mark.asyncio
async def test_flush_failure_before_the_sink_is_accounted_not_requeued(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(hook, "logger", fake_logger)
    delivered: list[dict] = []

    async def sink(records):
        delivered.extend(records)

    capture.register_capture_sink(sink)

    def broken_serialize(batch):
        raise ValueError("serializer exploded")

    monkeypatch.setattr(hook, "_serialize_batch", broken_serialize)
    capture.emit(KIND_SUMMARY_GENERATED, "lost", payload_kind="text")
    await capture.drain()

    # Re-queuing would pin a deterministic failure at the head of the buffer
    # forever; the batch is dropped and counted instead of vanishing silently.
    assert not hook._buffer
    assert hook._dropped == 1
    assert hook._in_flight_total() == 0
    assert delivered == []
    fake_logger.debug.assert_any_call("capture flush failed, %d event(s) dropped (%s)", 1, ANY)


@pytest.mark.asyncio
async def test_events_orphaned_by_clearing_the_sink_are_counted_as_dropped(fake_capture_sink):
    capture.emit(KIND_SUMMARY_GENERATED, "one", payload_kind="text")
    capture.emit(KIND_SUMMARY_GENERATED, "two", payload_kind="text")
    capture.register_capture_sink(None)

    await capture.drain()

    # Nothing to deliver to; the loss is accounted, not hidden.
    assert not hook._buffer
    assert fake_capture_sink.records == []
    assert hook._dropped == 2


@pytest.mark.asyncio
async def test_run_off_loop_does_not_retry_a_worker_failure_inline(monkeypatch):
    calls = 0

    def deep(_payload):
        nonlocal calls
        calls += 1
        raise RecursionError("maximum recursion depth exceeded")

    with pytest.raises(RecursionError):
        await run_off_loop(deep, "payload")
    # RecursionError is a RuntimeError subclass: a broad `except RuntimeError`
    # around the whole call would re-run the failing work on the event loop.
    assert calls == 1

    # The submit-time fallback (executor gone at interpreter exit) still works.
    def refuse(*_args, **_kwargs):
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", refuse)
    assert await run_off_loop(lambda x: x * 2, 21) == 42


# ---------------------------------------------------------------------------
# drain()'s budget is authoritative inside a batch; stranded loops never pin it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_deadline_bounds_sink_writes_inside_a_batch():
    calls: list[list[dict]] = []

    async def slow_sink(records):
        calls.append(records)
        await asyncio.sleep(1.0)  # every write takes 1 s (S3 under partition)

    capture.register_capture_sink(slow_sink)
    hook._configure(batch_size=64, flush_interval_s=60.0)  # SINK_TIMEOUT_S stays at 30 s

    # Three (run_id, kind) groups in ONE batch -> three sequential sink writes.
    for index in range(3):
        capture.emit(
            KIND_SUMMARY_GENERATED, f"g{index}", payload_kind="text", run_id=f"run-{index}"
        )

    started = time.monotonic()
    await capture.drain(timeout=0.2)
    elapsed = time.monotonic() - started

    # Checked only between batches, this would take 3 x min(1 s, SINK_TIMEOUT_S).
    assert elapsed < 0.6, elapsed
    assert len(calls) == 1  # the budget ran out during the first write
    # Nothing is lost: the cut-off group and the two never started are back, in order.
    assert [event.payload for event in hook._buffer] == ["g0", "g1", "g2"]
    assert hook._in_flight_total() == 0
    assert hook._dropped == 0


def test_a_batch_stranded_on_a_dead_loop_is_recovered_not_waited_for():
    async def wedged_sink(records):
        await asyncio.Event().wait()  # never returns

    capture.register_capture_sink(wedged_sink)
    hook._configure(batch_size=1, flush_interval_s=60.0)
    stranded = asyncio.new_event_loop()

    async def emit_and_return():
        capture.emit(KIND_SUMMARY_GENERATED, "stranded", payload_kind="text")  # BATCH_SIZE wake
        await asyncio.sleep(0.05)  # the flusher popped it and is parked inside the sink

    # run_until_complete returns with the flusher mid-write: the batch is stranded.
    stranded.run_until_complete(emit_and_return())
    [flusher] = hook._flushers.values()
    assert flusher.loop is stranded and not flusher.task.done()
    assert hook._in_flight_total() == 1

    # (1) Stopped, not closed: a fresh loop's drain must not wait for it.
    started = time.monotonic()
    asyncio.run(capture.drain(timeout=1.0))
    assert time.monotonic() - started < 0.5
    assert hook._in_flight_total() == 1  # still owned by the stranded loop, not lost

    # (2) Closed without cancelling (pytest-asyncio 0.21 style): the task died
    # with its loop, so its batch goes back to the buffer and the next drain
    # delivers it — at-least-once, never dropped — and still nobody waits for
    # the dead loop.
    delivered: list[dict] = []

    async def recording_sink(records):
        delivered.extend(records)

    capture.register_capture_sink(recording_sink)
    flusher.task._log_destroy_pending = False  # stranded on purpose; no GC noise
    stranded.close()
    started = time.monotonic()
    asyncio.run(capture.drain(timeout=1.0))
    assert time.monotonic() - started < 0.5
    assert [record["payload"] for record in delivered] == ["stranded"]
    assert hook._in_flight_total() == 0
    assert hook._dropped == 0
    assert not hook._buffer
    assert not hook._flushers


def test_thread_emits_wake_a_running_loop_not_a_stopped_one():
    got: list[dict] = []

    async def sink(records):
        got.extend(records)

    capture.register_capture_sink(sink)
    hook._configure(batch_size=4, flush_interval_s=60.0)
    stale = asyncio.new_event_loop()

    async def open_scope_only():
        with capture.run_scope("stale", kind="pipeline"):
            pass  # scope entry started a flusher on this loop

    stale.run_until_complete(open_scope_only())
    # Stopped, not closed: it stays registered, first in insertion order.
    assert next(iter(hook._flushers)) is stale

    async def live_run():
        with capture.run_scope("live", kind="pipeline"):

            def worker():
                for index in range(8):
                    capture.emit(KIND_SUMMARY_GENERATED, f"t{index}", payload_kind="text")

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            # No drain and a 60 s interval: only the BATCH_SIZE wake can deliver
            # these, and only if it reached THIS loop's flusher, not the stale one
            # (a wake scheduled on a stopped loop never runs).
            await _wait_until(lambda: len(got) >= 8, timeout=1.0)

    try:
        asyncio.run(live_run())
        assert hook._flushers[stale].wake_pending is False
    finally:
        stale_flusher = hook._flushers.get(stale)
        if stale_flusher is not None:
            hook._cancel_flusher(stale_flusher, wait=True)
        stale.close()


def test_flusher_burst_loop_terminates_when_a_batch_pops_nothing(monkeypatch, fake_capture_sink):
    # Bypass the clamp on purpose: this is the shape a misconfigured (or raced)
    # flush takes — _flush_one_batch pops nothing and returns without suspending.
    monkeypatch.setattr(hook, "BATCH_SIZE", 0)
    finished = threading.Event()
    outcome: dict = {}

    async def scenario():
        capture.emit(KIND_SUMMARY_GENERATED, "x", payload_kind="text")  # the wake fires at once
        # A flusher spinning without yielding would never let this sleep return.
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await capture.drain(timeout=0.5)
        outcome["drain_s"] = time.monotonic() - started
        outcome["buffered"] = len(hook._buffer)

    def runner():
        try:
            asyncio.run(scenario())
        finally:
            finished.set()

    threading.Thread(target=runner, daemon=True).start()
    # Off-thread with a hard wait so a regression fails instead of hanging pytest.
    assert finished.wait(timeout=5.0), "the flusher monopolised its event loop"
    assert outcome["drain_s"] < 0.2  # stopped on the empty pop, not at its deadline
    assert outcome["buffered"] == 1  # inert, not fatal: nothing delivered, nothing lost


def test_configure_clamps_degenerate_knobs():
    hook._configure(queue_size=0, batch_size=-1, flush_interval_s=0.0)

    assert hook.QUEUE_SIZE == 1
    assert hook.BATCH_SIZE == 1
    assert hook.FLUSH_INTERVAL_S == hook._MIN_FLUSH_INTERVAL_S


# ---------------------------------------------------------------------------
# atexit: the CLI story end to end (asyncio.run per command, no drain)
# ---------------------------------------------------------------------------


def test_atexit_hook_persists_leftovers_after_asyncio_run(tmp_path):
    script = textwrap.dedent(
        f"""
        import asyncio

        from cognee.infrastructure.files.storage import StorageManager
        from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
        from cognee.modules.observability import capture
        from cognee.modules.observability.capture import hook

        storage = StorageManager(LocalFileStorage({str(tmp_path)!r}))
        capture.register_capture_sink(capture.StorageSink(storage))
        hook._configure(flush_interval_s=60.0)

        async def command():
            # One asyncio.run per CLI command and no drain: the flusher is
            # asleep on its interval when asyncio.run tears the loop down.
            with capture.run_scope("run-1", "ds-1", kind="pipeline"):
                for index in range(3):
                    capture.emit(
                        capture.KIND_SUMMARY_GENERATED, f"s{{index}}", payload_kind="text"
                    )

        asyncio.run(command())
        # 3 events + the manifest are still buffered; only the atexit hook remains.
        assert len(hook._buffer) == 4, len(hook._buffer)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env={**os.environ, "TELEMETRY_DISABLED": "1"},
    )
    assert result.returncode == 0, result.stderr

    run_dir = tmp_path / "ds-1" / "run-1"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload"]["kind"] == "pipeline"
    assert manifest["payload"]["dropped_events"] == 0
    [blob] = (run_dir / KIND_SUMMARY_GENERATED).glob("batch-*.jsonl.gz")
    with gzip.open(blob, "rt", encoding="utf-8") as lines:
        payloads = [json.loads(line)["payload"] for line in lines if line.strip()]
    assert payloads == ["s0", "s1", "s2"]


@pytest.mark.asyncio
async def test_drain_is_bounded_even_when_a_sink_swallows_its_cancellation():
    """A sink that ignores CancelledError must not pin drain() forever.

    Sinks are contractually required to let the cancel propagate, but
    register_capture_sink is public API. Before the _CANCEL_GRACE_S bound the
    post-cancel wait was unbounded, so drain() ran past its budget without limit
    and no outer wait_for could rescue the caller (the CancelledError branch had
    the same unbounded wait) — hanging run_tasks, which awaits drain() inline.
    """

    async def uncancellable(records):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(30)  # swallows the cancel

    capture.register_capture_sink(uncancellable)
    hook._configure(flush_interval_s=60.0, sink_timeout_s=0.1)

    capture.emit(KIND_SUMMARY_GENERATED, "wedged", payload_kind="text")

    started = time.monotonic()
    await capture.drain(timeout=0.2)
    elapsed = time.monotonic() - started

    # The budget plus at most one cancellation grace, not 30s.
    assert elapsed < 0.2 + hook._CANCEL_GRACE_S + 1.0

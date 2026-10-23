"""Resource/failure boundaries of the real batching sender; no external telemetry."""

import asyncio
import gc
import json
import threading
import time
import tracemalloc
from dataclasses import replace

import httpx
import pytest
from tenacity import wait_none

from cognee.shared import telemetry_sender, utils
from cognee.shared.telemetry_limits import (
    CLOUD_EVENTS,
    PayloadTooLarge,
    TelemetryLimits,
    bounded_json,
)
from cognee.shared.telemetry_sender import TelemetrySender


def wait_for(condition):
    deadline = time.monotonic() + 3
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.fixture
def make_sender(monkeypatch):
    monkeypatch.setattr(telemetry_sender, "FLUSH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(telemetry_sender, "wait_exponential_jitter", lambda **_: wait_none())
    senders = []

    def make(handler, **kwargs):
        sender = TelemetrySender("http://proxy.test", 0.1, httpx.MockTransport(handler), **kwargs)
        senders.append(sender)
        return sender

    yield make
    for sender in senders:
        sender.close()


@pytest.mark.parametrize("value", ["a" * 100_000, "😀" * 50, list(range(10_000))])
def test_oversize_is_refused_before_network(make_sender, value):
    calls = []
    sender = make_sender(lambda r: calls.append(r), limits=TelemetryLimits(event_bytes=128))
    sender.enqueue({"value": value})
    assert sender.snapshot()["rejected_size"] == 1
    assert sender.snapshot()["queue_bytes"] == 0
    assert calls == []
    assert sender._thread is None


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), 2**1000, {1: "key"}])
def test_invalid_data_is_refused(make_sender, value):
    sender = make_sender(lambda _: pytest.fail("must not send"))
    sender.enqueue({"value": value})
    assert sender.snapshot()["rejected_format"] == 1


def test_cyclic_or_deep_input_is_bounded():
    recursive = []
    recursive.append(recursive)
    with pytest.raises(PayloadTooLarge):
        bounded_json(recursive, 16_384)
    with pytest.raises(PayloadTooLarge):
        bounded_json([None] * 3000, 16_384)


def test_byte_batches_preserve_snapshot_and_framing(make_sender):
    bodies = []
    limits = TelemetryLimits(event_bytes=128, batch_bytes=192)
    sender = make_sender(
        lambda r: (bodies.append(r.content), httpx.Response(200))[1], limits=limits
    )
    payload = {"value": "a" * 80}
    with sender._condition:
        for _ in range(5):
            sender.enqueue(payload)
    payload["value"] = "changed after enqueue"
    sender.close()
    assert len(bodies) == 3
    assert all(len(body) <= limits.batch_bytes for body in bodies)
    assert [event for body in bodies for event in json.loads(body)] == [{"value": "a" * 80}] * 5
    assert sender.snapshot()["peak_inflight_bytes"] <= 2 * limits.batch_bytes


def test_sustained_burst_with_stalled_proxy_bounds_memory_and_concurrency(make_sender):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def stalled(request):
        calls.append(request.content)
        entered.set()
        assert release.wait(3)
        return httpx.Response(503)

    limits = TelemetryLimits(
        event_bytes=256,
        batch_bytes=512,
        queue_bytes=1024,
        events_per_minute=100_000,
        bytes_per_minute=10_000_000,
    )
    sender = make_sender(stalled, limits=limits)
    payload = {"value": "x" * 200}
    sender.enqueue(payload)
    assert entered.wait(1)
    try:
        started = time.monotonic()
        for _ in range(10_000):
            sender.enqueue(payload)
        assert time.monotonic() - started < 2
        state = sender.snapshot()
        assert state["queue_bytes"] <= 1024
        assert state["inflight_bytes"] <= 1024
        assert state["peak_queue_bytes"] <= 1024
        assert state["dropped_queue"] > 9900
        assert len(calls) == 1
        worker = sender._thread
        started = time.monotonic()
        sender.close()
        assert time.monotonic() - started < 0.5
        sender.enqueue(payload)
        assert sender._thread is worker
        assert sender.snapshot()["queue_bytes"] == 0
        assert sender.snapshot()["dropped_shutdown"] > 0
    finally:
        release.set()
        wait_for(lambda: sender._thread is None)
    assert len(calls) == 1  # No late retry or drain after the close deadline.


@pytest.mark.parametrize("status", [400, 401, 403, 413, 422, 429])
def test_rejection_never_fans_out_or_counts_as_exported(make_sender, status):
    bodies = []

    def reject(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(status)

    sender = make_sender(reject)
    with sender._condition:
        for n in range(10):
            sender.enqueue({"n": n})
    wait_for(lambda: sender.snapshot()["dropped_http"] == 10)
    sender.enqueue({"n": 11})
    sender.close()
    assert [len(body) for body in bodies] == [10, 1]
    assert sender.snapshot()["exported_events"] == 0
    assert sender.snapshot()["legacy_fallbacks"] == 0


@pytest.mark.parametrize("failure", [503, httpx.ReadTimeout("timeout"), httpx.ConnectError("down")])
def test_failures_stop_after_bounded_attempts(make_sender, failure):
    calls = []

    def unavailable(request):
        calls.append(request.content)
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure)

    sender = make_sender(unavailable)
    sender.enqueue({"n": 1})
    wait_for(lambda: sender.snapshot()["inflight_bytes"] == 0 and len(calls) == 4)
    assert sender.snapshot()["exported_events"] == 0
    assert sender.snapshot()["retries"] == 3
    assert all(body == calls[0] for body in calls)


def test_request_budget_also_caps_legacy_fallback(make_sender):
    bodies = []

    def legacy(request):
        body = json.loads(request.content)
        bodies.append(body)
        return (
            httpx.Response(422, headers={"X-Cognee-Telemetry-Batch": "unsupported"})
            if isinstance(body, list)
            else httpx.Response(200)
        )

    sender = make_sender(legacy, limits=TelemetryLimits(requests_per_minute=3))
    with sender._condition:
        for n in range(20):
            sender.enqueue({"event_name": "cognee.recall", "n": n})
    sender.close()
    assert len(bodies) == 3  # One probe, at most two singles, not 21 requests.
    assert sender.snapshot()["exported_events"] == 2
    assert sender.snapshot()["legacy_fallbacks"] == 1


def test_operator_can_explicitly_select_old_proxy(make_sender, monkeypatch):
    monkeypatch.setenv("TELEMETRY_LEGACY_PROXY", "true")
    bodies = []
    sender = make_sender(lambda r: (bodies.append(json.loads(r.content)), httpx.Response(200))[1])
    sender.enqueue({"n": 1})
    sender.close()
    assert bodies == [{"n": 1}]


def test_rate_limits_and_refill_without_sleeping(make_sender):
    bodies = []
    sender = make_sender(
        lambda r: (bodies.append(r.content), httpx.Response(200))[1],
        limits=TelemetryLimits(events_per_minute=2),
    )
    with sender._condition:
        for n in range(5):
            sender.enqueue({"event_name": "cognee.recall", "n": n})
    sender.close()
    assert sender.snapshot()["exported_events"] == 2
    assert sender.snapshot()["dropped_rate"] == 3
    # Reopening doesn't reset the rate budget. A minute of monotonic time refills it.
    sender.enqueue({"event_name": "cognee.recall", "n": 6})
    assert sender.snapshot()["dropped_rate"] == 4
    sender._events.updated -= 60
    sender.enqueue({"event_name": "cognee.recall", "n": 7})
    sender.close()
    assert sender.snapshot()["exported_events"] == 3


def test_wire_byte_budget_counts_retries(make_sender):
    calls = []
    sender = make_sender(
        lambda r: (calls.append(r.content), httpx.Response(503))[1],
        limits=TelemetryLimits(bytes_per_minute=100),
    )
    sender.enqueue({"event_name": "cognee.recall", "value": "x" * 40})
    wait_for(lambda: sender.snapshot()["dropped_rate"] == 1)
    assert len(calls) == 1  # The same payload cannot spend the budget four times.
    assert sender.snapshot()["request_bytes"] <= 100


def test_response_body_is_never_buffered(make_sender):
    class EndlessBody(httpx.SyncByteStream):
        def __iter__(self):
            pytest.fail("the exporter must not consume response content")
            yield b""

    sender = make_sender(lambda _: httpx.Response(200, stream=EndlessBody()))
    sender.enqueue({"n": 1})
    sender.close()
    assert sender.snapshot()["exported_events"] == 1


def test_fork_reset_discards_inherited_state_and_locks(make_sender):
    sender = make_sender(lambda _: httpx.Response(200))
    sender._preparation.acquire()
    sender.count("dropped_queue", 20)
    sender._reset()  # Same callback registered with os.register_at_fork.
    assert sender._preparation.acquire(blocking=False)
    sender._preparation.release()
    assert sender.snapshot()["dropped_queue"] == 0
    assert sender.snapshot()["queue_bytes"] == 0


def test_counters_are_fixed_and_saturating(make_sender):
    sender = make_sender(lambda _: httpx.Response(200))
    before = set(sender.snapshot())
    sender.count("tenant-secret-uuid", 42)
    sender.count("dropped_queue", 2**100)
    sender.count("dropped_queue")
    assert set(sender.snapshot()) == before
    assert sender.snapshot()["dropped_queue"] == 2**63 - 1


@pytest.mark.parametrize("value", ["0", "-1", "broken", str(2**60)])
def test_invalid_environment_limits_remain_bounded(monkeypatch, value):
    monkeypatch.setenv("TELEMETRY_EVENT_BYTES", value)
    assert TelemetryLimits.from_env() == TelemetryLimits()


@pytest.fixture
def cloud_sender(make_sender, monkeypatch):
    payloads = []
    sender = make_sender(lambda r: (payloads.extend(json.loads(r.content)), httpx.Response(200))[1])
    monkeypatch.setattr(utils, "_telemetry_sender", sender)
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("TELEMETRY_ORIGIN", "cloud")
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(utils, "get_anonymous_id", lambda: "tenant")
    monkeypatch.setattr(utils, "get_persistent_id", lambda: "test")
    monkeypatch.setattr(utils, "_get_api_key_tracking_id", lambda: "")
    return sender, payloads


def test_cloud_selection_keeps_correlated_outcomes_and_drops_details(cloud_sender):
    sender, payloads = cloud_sender
    for name in (
        "Pipeline Run Started",
        "Pipeline Run Errored",
        "cognee.recall",
        "cognee.recall ERRORED",
    ):
        utils.send_telemetry(
            name, additional_properties={"pipeline_run_id": "same-run", "telemetry_origin": "sdk"}
        )
    for name in (
        "Pipeline Item Started",
        "extract_graph Task Completed",
        "Token Usage",
        "new unknown event",
    ):
        utils.send_telemetry(name)
    sender.close()
    assert len(payloads) == 4
    assert all(p["properties"]["pipeline_run_id"] == "same-run" for p in payloads)
    assert all(p["properties"]["telemetry_origin"] == "cloud" for p in payloads)
    assert sender.snapshot()["sampled_events"] == 4


def test_cloud_filter_runs_before_identity_and_property_work(cloud_sender, monkeypatch):
    sender, _ = cloud_sender
    monkeypatch.setattr(utils, "get_persistent_id", lambda: pytest.fail("identity lookup"))
    utils.send_telemetry("Pipeline Item Started", additional_properties={"huge": "x" * 1_000_000})
    assert sender.snapshot()["sampled_events"] == 1


def test_payload_preparation_failure_does_not_escape(cloud_sender, monkeypatch):
    sender, _ = cloud_sender

    def broken():
        raise RuntimeError("identity lookup failed")

    monkeypatch.setattr(utils, "get_persistent_id", broken)
    assert utils.send_telemetry("Pipeline Run Started") is None
    assert sender.snapshot()["dropped_internal"] == 1


def test_excess_preparation_waits_only_for_a_bounded_time(cloud_sender):
    sender, _ = cloud_sender
    for _ in range(telemetry_sender.PREPARATION_CONCURRENCY):
        sender._preparation.acquire()
    try:
        started = time.monotonic()
        utils.send_telemetry("Pipeline Run Started")
        assert telemetry_sender.ADMISSION_WAIT_SECONDS <= time.monotonic() - started < 0.2
        assert sender.snapshot()["dropped_busy"] == 1
    finally:
        for _ in range(telemetry_sender.PREPARATION_CONCURRENCY):
            sender._preparation.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("original"), asyncio.CancelledError()])
async def test_export_failure_cannot_change_product_error_or_cancellation(cloud_sender, error):
    sender, _ = cloud_sender
    # Force exporter rejection at the product boundary.
    for _ in range(telemetry_sender.PREPARATION_CONCURRENCY):
        sender._preparation.acquire()

    @utils.telemetry_on_error("cognee.recall ERRORED")
    async def operation():
        raise error

    try:
        with pytest.raises(type(error)) as result:
            await operation()
        assert result.value is error
    finally:
        for _ in range(telemetry_sender.PREPARATION_CONCURRENCY):
            sender._preparation.release()


def test_serialization_does_not_retain_buffers_without_cyclic_gc():
    payload = {"value": "x" * 15_000}
    was_enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    try:
        for _ in range(2000):
            bounded_json(payload, 16_384)
        current, peak = tracemalloc.get_traced_memory()
        assert current < 64 * 1024
        assert peak < 128 * 1024
    finally:
        tracemalloc.stop()
        if was_enabled:
            gc.enable()


_REAL_TRACKING_ID = utils._get_api_key_tracking_id


def test_four_simultaneous_product_events_survive_cold_key_derivation(cloud_sender, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    sender, payloads = cloud_sender
    monkeypatch.setattr(utils, "_get_api_key_tracking_id", _REAL_TRACKING_ID)
    monkeypatch.setenv("LLM_API_KEY", "synthetic-concurrent-test-key")
    barrier = threading.Barrier(4)

    def emit(index):
        barrier.wait(timeout=2)
        utils.send_telemetry("cognee.recall", additional_properties={"call_index": index})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(emit, range(4)))
    sender.close()
    assert sorted(p["properties"]["call_index"] for p in payloads) == list(range(4))
    assert sender.snapshot()["dropped_busy"] == 0


def test_sdk_task_burst_cannot_spend_run_outcome_budget(make_sender):
    payloads = []
    sender = make_sender(lambda r: (payloads.extend(json.loads(r.content)), httpx.Response(200))[1])
    with sender._condition:
        sender.enqueue({"event_name": "Pipeline Run Started", "pipeline_run_id": "run"})
        for n in range(240):
            sender.enqueue({"event_name": "extract_graph Task Completed", "n": n})
        sender.enqueue({"event_name": "Pipeline Run Completed", "pipeline_run_id": "run"})
    sender.close()
    assert payloads[0]["event_name"] == "Pipeline Run Started"
    assert payloads[-1]["event_name"] == "Pipeline Run Completed"
    assert sender.snapshot()["exported_events"] <= 120
    assert sender.snapshot()["dropped_rate"] > 0


def test_cloud_allowlist_contains_only_wired_emitters():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    emitters = [path for path in root.rglob("*.py") if "tests" not in path.relative_to(root).parts]
    nodes = [node for path in emitters for node in ast.walk(ast.parse(path.read_text()))]
    constants = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants.setdefault(target.id, set()).add(node.value.value)
    emitted = set()
    for node in nodes:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("send_telemetry", "telemetry_on_error")
            and node.args
        ):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            emitted.add(first.value)
        elif isinstance(first, ast.Name):
            emitted.update(constants.get(first.id, ()))
    assert not (CLOUD_EVENTS - emitted), f"Unwired cloud events: {CLOUD_EVENTS - emitted}"


@pytest.mark.parametrize("budget", ["logical_bytes", "wire_bytes", "requests"])
def test_details_cannot_spend_reserved_run_capacity(make_sender, budget, monkeypatch):
    payloads = []
    sender = make_sender(lambda r: (payloads.extend(json.loads(r.content)), httpx.Response(200))[1])
    with sender._condition:
        bucket, reserve = {
            "logical_bytes": (sender._bytes, sender._byte_reserve),
            "wire_bytes": (sender._wire_bytes, sender._byte_reserve),
            "requests": (sender._requests, sender._request_reserve),
        }[budget]
        bucket.tokens = reserve
        # Freeze refill: even a 10 ms thread start can refill one small event.
        available = bucket.available
        monkeypatch.setattr(
            bucket, "available", lambda amount, now: available(amount, bucket.updated)
        )
        sender.enqueue({"event_name": "Pipeline Item Errored"})
        sender.enqueue({"event_name": "Pipeline Run Started"})
        sender.enqueue({"event_name": "Pipeline Run Completed"})
    sender.close()
    assert [p["event_name"] for p in payloads] == ["Pipeline Run Started", "Pipeline Run Completed"]
    assert sender.snapshot()["dropped_rate"] == 1
    assert sender.snapshot()["request_bytes"] <= sender.limits.bytes_per_minute


def test_full_queue_evicts_newest_details_for_run_outcome(make_sender):
    payloads = []
    limits = TelemetryLimits(event_bytes=128, queue_bytes=160, batch_bytes=256)
    sender = make_sender(
        lambda r: (payloads.extend(json.loads(r.content)), httpx.Response(200))[1], limits=limits
    )
    with sender._condition:
        sender.enqueue({"event_name": "Pipeline Run Started"})
        sender.enqueue({"event_name": "task", "padding": "x" * 20, "n": 1})
        sender.enqueue({"event_name": "task", "padding": "x" * 20, "n": 2})
        sender.enqueue({"event_name": "Pipeline Run Completed"})
    sender.close()
    assert [p["event_name"] for p in payloads] == [
        "Pipeline Run Started",
        "task",
        "Pipeline Run Completed",
    ]
    assert payloads[1]["n"] == 1
    assert sender.snapshot()["dropped_queue"] == 1
    assert sender.snapshot()["peak_queue_bytes"] <= limits.queue_bytes


def test_run_only_overload_remains_bounded(make_sender):
    limits = TelemetryLimits(event_bytes=128, queue_bytes=128, batch_bytes=256)
    sender = make_sender(lambda _: httpx.Response(200), limits=limits)
    with sender._condition:
        for _ in range(10):
            sender.enqueue({"event_name": "Pipeline Run Completed"})
    sender.close()
    assert sender.snapshot()["dropped_queue"] > 0
    assert sender.snapshot()["peak_queue_bytes"] <= limits.queue_bytes


def test_cloud_keeps_item_error_context_but_not_endpoint_census(cloud_sender):
    sender, payloads = cloud_sender
    utils.send_telemetry(
        "Pipeline Item Errored",
        additional_properties={
            "pipeline_run_id": "run",
            "loader": "TextLoader",
            "extension": ".txt",
        },
    )
    utils.send_telemetry("Cognify API Endpoint Invoked")
    sender.close()
    assert [p["event_name"] for p in payloads] == ["Pipeline Item Errored"]
    assert payloads[0]["properties"]["loader"] == "TextLoader"
    assert sender.snapshot()["sampled_events"] == 1


def test_sanitizable_raw_url_can_exceed_wire_event_cap(cloud_sender):
    sender, payloads = cloud_sender
    raw_url = "https://synthetic.invalid/" + "x" * 32_000
    utils.send_telemetry("cognee.recall", additional_properties={"url": raw_url})
    sender.close()
    assert len(payloads) == 1
    assert len(payloads[0]["properties"]["url"]) < 100
    assert raw_url not in str(payloads)
    assert sender.snapshot()["rejected_size"] == 0


@pytest.mark.parametrize("key,length", [("padding", 32_000), ("url", 100_000)])
def test_raw_and_final_property_sizes_stay_bounded(cloud_sender, key, length):
    sender, payloads = cloud_sender
    utils.send_telemetry("cognee.recall", additional_properties={key: "x" * length})
    sender.close()
    assert payloads == []
    assert sender.snapshot()["rejected_size"] == 1


def test_tracking_id_cache_reuses_warm_value_and_respects_rotation(monkeypatch):
    import hashlib

    original = hashlib.pbkdf2_hmac
    calls = []

    def derive(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", derive)
    monkeypatch.setenv("LLM_API_KEY", "synthetic-cache-test-key")
    monkeypatch.setenv("TELEMETRY_API_KEY_TRACKING_SALT", "first-salt")
    utils._derive_api_key_tracking_id.cache_clear()
    try:
        first = _REAL_TRACKING_ID()
        assert _REAL_TRACKING_ID() == first
        assert len(calls) == 1
        monkeypatch.setenv("LLM_API_KEY", "synthetic-rotated-key")
        second = _REAL_TRACKING_ID()
        assert second != first
        monkeypatch.setenv("TELEMETRY_API_KEY_TRACKING_SALT", "second-salt")
        assert _REAL_TRACKING_ID() != second
        assert len(calls) == 3
        assert utils._derive_api_key_tracking_id.cache_info().currsize == 1
    finally:
        utils._derive_api_key_tracking_id.cache_clear()


def test_sender_instances_do_not_accumulate_process_hooks(monkeypatch):
    import weakref

    monkeypatch.setattr(telemetry_sender.atexit, "register", lambda *_: pytest.fail("extra hook"))
    monkeypatch.setattr(
        telemetry_sender.os, "register_at_fork", lambda **_: pytest.fail("extra hook")
    )
    references = []
    for _ in range(10):
        sender = TelemetrySender("http://proxy.test", 0.1)
        references.append(weakref.ref(sender))
    del sender
    gc.collect()
    assert all(reference() is None for reference in references)

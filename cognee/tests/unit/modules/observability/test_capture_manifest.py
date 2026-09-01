"""run_scope / RunScope manifest contract and per-run sampling (SDK-529)."""

import asyncio
from uuid import uuid4

import pytest

from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_RETRIEVAL_CANDIDATES,
    KIND_RUN_MANIFEST,
    KIND_SUMMARY_GENERATED,
    hook,
)

pytestmark = pytest.mark.usefixtures("capture_reset")


def _manifests(sink):
    return [record for record in sink.records if record["kind"] == KIND_RUN_MANIFEST]


# ---------------------------------------------------------------------------
# 7: manifest contents, late-bound dataset, child-task accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_carries_fields_counters_and_timing(fake_capture_sink):
    run_id = uuid4()
    dataset_id = uuid4()

    with capture.run_scope(run_id, dataset_id, kind="pipeline") as scope:
        assert capture.current_scope() is scope
        capture.note("llm_model", "gpt-5-mini")
        capture.bump("chunks")
        capture.bump("chunks", 2)
        scope.note("prompt", "sha256:abc")
        scope.bump("nodes", 5)
    assert capture.current_scope() is None

    await capture.drain()

    [manifest] = _manifests(fake_capture_sink)
    assert manifest["run_id"] == str(run_id)
    assert manifest["dataset_id"] == str(dataset_id)
    payload = manifest["payload"]
    assert payload["run_id"] == str(run_id)
    assert payload["dataset_id"] == str(dataset_id)
    assert payload["kind"] == "pipeline"
    assert payload["sampled"] is True
    assert payload["llm_model"] == "gpt-5-mini"
    assert payload["prompt"] == "sha256:abc"
    assert payload["counters"] == {"chunks": 3, "nodes": 5}
    assert payload["ended_at"] >= payload["started_at"]
    assert payload["duration_s"] >= 0
    assert payload["dropped_events"] == 0


@pytest.mark.asyncio
async def test_set_dataset_late_binds_already_buffered_events(fake_capture_sink):
    run_id = uuid4()
    dataset_id = uuid4()

    with capture.run_scope(run_id, kind="operation") as scope:
        capture.emit(KIND_SUMMARY_GENERATED, "before", payload_kind="text")
        assert hook._buffer[0].dataset_id is None
        scope.set_dataset(dataset_id)

    await capture.drain()

    records = fake_capture_sink.records
    assert len(records) == 2
    assert all(record["run_id"] == str(run_id) for record in records)
    assert all(record["dataset_id"] == str(dataset_id) for record in records)
    [manifest] = _manifests(fake_capture_sink)
    assert manifest["payload"]["dataset_id"] == str(dataset_id)


@pytest.mark.asyncio
async def test_explicit_event_ids_override_scope(fake_capture_sink):
    override_run = uuid4()

    with capture.run_scope(uuid4(), uuid4(), kind="pipeline"):
        capture.emit(KIND_SUMMARY_GENERATED, "x", payload_kind="text", run_id=override_run)

    await capture.drain()

    [record] = [r for r in fake_capture_sink.records if r["kind"] == KIND_SUMMARY_GENERATED]
    assert record["run_id"] == str(override_run)


@pytest.mark.asyncio
async def test_child_task_bump_reaches_parent_manifest(fake_capture_sink):
    async def child():
        capture.bump("child_hits")
        capture.note("from_child", True)

    with capture.run_scope(uuid4(), kind="pipeline"):
        await asyncio.gather(asyncio.create_task(child()), asyncio.create_task(child()))

    await capture.drain()

    [manifest] = _manifests(fake_capture_sink)
    assert manifest["payload"]["counters"] == {"child_hits": 2}
    assert manifest["payload"]["from_child"] is True


def test_nested_scopes_restore_the_parent():
    with capture.run_scope("outer", kind="pipeline") as outer:
        with capture.run_scope("inner", kind="pipeline") as inner:
            assert inner.parent is outer
            assert capture.current_scope() is inner
        assert capture.current_scope() is outer
    assert capture.current_scope() is None


@pytest.mark.asyncio
async def test_manifest_is_emitted_even_when_the_body_raises(fake_capture_sink):
    with pytest.raises(RuntimeError):
        with capture.run_scope(uuid4(), kind="pipeline"):
            raise RuntimeError("body failed")

    await capture.drain()
    assert len(_manifests(fake_capture_sink)) == 1


@pytest.mark.asyncio
async def test_manifest_survives_a_full_buffer_and_reports_the_drops(fake_capture_sink):
    hook._configure(queue_size=2, flush_interval_s=60.0)
    run_id = uuid4()

    with capture.run_scope(run_id, uuid4(), kind="pipeline"):
        for index in range(5):
            capture.emit(KIND_SUMMARY_GENERATED, f"s{index}", payload_kind="text")
        assert hook._dropped == 3
        assert len(hook._buffer) == 2
    # The manifest gets headroom past the bound: it is the record that reports the drops.
    assert len(hook._buffer) == 3

    await capture.drain()

    [manifest] = _manifests(fake_capture_sink)
    assert manifest["run_id"] == str(run_id)
    assert manifest["payload"]["dropped_events"] == 3


def test_manifests_are_bounded_past_queue_size(fake_capture_sink):
    # A sync caller with no loop anywhere: nothing can flush, so every completed
    # run's manifest stays buffered. Manifests bypass QUEUE_SIZE, but not 2x it.
    hook._configure(queue_size=4)

    for index in range(50):
        with capture.run_scope(f"run-{index}", kind="pipeline"):
            pass

    assert len(hook._buffer) == 2 * hook.QUEUE_SIZE
    assert hook._dropped == 50 - 2 * hook.QUEUE_SIZE
    assert not hook._flushers


@pytest.mark.asyncio
async def test_finish_emits_the_manifest_once_and_exit_skips_the_duplicate(fake_capture_sink):
    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        capture.bump("items", 2)
        assert scope.finished is False
        scope.finish()
        assert scope.finished is True
        # A caller that drains before its scope closes sees its own manifest.
        await capture.drain()
        [manifest] = _manifests(fake_capture_sink)
        assert manifest["payload"]["counters"] == {"items": 2}
        scope.finish()  # idempotent

    await capture.drain()
    assert len(_manifests(fake_capture_sink)) == 1


@pytest.mark.asyncio
async def test_note_cannot_overwrite_envelope_keys(fake_capture_sink):
    run_id = uuid4()

    with capture.run_scope(run_id, kind="pipeline"):
        capture.note("run_id", "spoofed")
        capture.note("kind", "spoofed")
        capture.note("counters", "not a dict")
        capture.note("model", "gpt-5-mini")
        capture.bump("hits")

    await capture.drain()

    [manifest] = _manifests(fake_capture_sink)
    payload = manifest["payload"]
    assert manifest["run_id"] == str(run_id)  # filed under the real run
    assert payload["run_id"] == str(run_id)
    assert payload["kind"] == "pipeline"
    assert payload["counters"] == {"hits": 1}
    assert payload["model"] == "gpt-5-mini"


# ---------------------------------------------------------------------------
# 8: sampling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_rate_zero_unsamples_operation_scopes_only(fake_capture_sink):
    hook._configure(sample_rate=0.0)

    with capture.run_scope(uuid4(), kind="operation") as scope:
        assert scope.sampled is False
        assert capture.should_capture(KIND_RETRIEVAL_CANDIDATES) is False
        assert capture.should_capture(KIND_EXTRACTION_CHUNK_GRAPH) is True
    await capture.drain()
    assert _manifests(fake_capture_sink) == []

    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        assert scope.sampled is True
        assert capture.should_capture(KIND_RETRIEVAL_CANDIDATES) is True
    await capture.drain()
    assert len(_manifests(fake_capture_sink)) == 1

    # No active scope: fall back to the rate.
    assert capture.should_capture(KIND_RETRIEVAL_CANDIDATES) is False
    assert capture.should_capture(KIND_SUMMARY_GENERATED) is True


@pytest.mark.asyncio
async def test_sample_rate_one_keeps_everything(fake_capture_sink):
    hook._configure(sample_rate=1.0)

    for _ in range(5):
        with capture.run_scope(uuid4(), kind="operation") as scope:
            assert scope.sampled is True
            assert capture.should_capture(KIND_RETRIEVAL_CANDIDATES) is True
    assert capture.should_capture(KIND_RETRIEVAL_CANDIDATES) is True

    await capture.drain()
    assert len(_manifests(fake_capture_sink)) == 5

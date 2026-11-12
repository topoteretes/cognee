import pytest

from cognee.infrastructure.databases.unified import batch_writer
from cognee.infrastructure.databases.unified.storage_trace import (
    collect_storage_events,
    storage_batch,
    storage_span,
)


def test_registry_only_selects_supported_pair_and_rejects_ambiguity(monkeypatch):
    monkeypatch.setattr(batch_writer, "_factories", {})
    graph, vector, writer = object(), object(), object()
    factory = lambda g, v: writer if g is graph and v is vector else None
    batch_writer.register_batch_writer_factory("test", factory)
    batch_writer.register_batch_writer_factory("test", factory)
    assert batch_writer.get_batch_writer(graph, vector) is writer
    assert batch_writer.get_batch_writer(graph, object()) is None
    batch_writer.register_batch_writer_factory("conflict", factory)
    with pytest.raises(RuntimeError, match="Multiple batch writers"):
        batch_writer.get_batch_writer(graph, vector)


def test_timeline_has_batch_identity_and_failure_without_payloads():
    events = []
    with collect_storage_events(events.append), storage_batch("Point_text", 128):
        with storage_span("prepare"):
            pass
        with pytest.raises(ValueError), storage_span("merge"):
            raise ValueError("write failed")
    assert len(events) == 2
    assert events[0]["batch_id"] == events[1]["batch_id"]
    assert events[0]["ok"] is True and events[1]["ok"] is False
    assert all(event["elapsed_s"] >= 0 for event in events)
    assert set(events[0]) == {
        "batch_id",
        "collection",
        "rows",
        "stage",
        "start_s",
        "elapsed_s",
        "ok",
    }


def test_telemetry_failure_does_not_change_write_outcome():
    def broken_sink(event):
        raise RuntimeError("telemetry offline")

    with collect_storage_events(broken_sink), storage_batch("Point_text", 1):
        with storage_span("merge"):
            pass
        with pytest.raises(ValueError, match="original"), storage_span("merge"):
            raise ValueError("original")

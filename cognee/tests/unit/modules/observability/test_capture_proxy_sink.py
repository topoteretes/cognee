"""ProxySink (SDK-529): what leaves the process on the way to the analytics proxy —
metadata by default, the identity telemetry sends, masked model paths — in
byte-capped batches, off the event loop, and only when the telemetry kill
switches allow it. No network: the poster is a recorder."""

import asyncio
import json
import time
from uuid import uuid4

import pytest

from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RETRIEVAL_CANDIDATES,
    KIND_RUN_MANIFEST,
    KIND_SETUP_CHANGED,
    KIND_STORAGE_DELTA,
    KIND_SUMMARY_GENERATED,
    CaptureConfig,
    FanOutSink,
    ProxySink,
    StorageSink,
    hook,
)
from cognee.modules.observability.capture import events as events_module
from cognee.modules.observability.capture import proxy_sink as proxy_sink_module
from cognee.modules.observability.capture.proxy_sink import (
    CONTENT_KINDS,
    default_proxy_url,
    prepare_record,
    proxy_sink_blocked,
)
from cognee.tests.unit.modules.conftest import FakeCaptureSink

pytestmark = pytest.mark.usefixtures("capture_reset")

IDENTITY = {
    "anonymous_id": "anon-id",
    "persistent_id": "persistent-id",
    "api_key_tracking_id": "ak_test",
    "telemetry_origin": "sdk",
    "cognee_version": "0.0.0-test",
    "sdk_runtime": "python",
}

CHUNK_GRAPH = {
    "chunk_id": "chunk-1",
    "chunk_index": 0,
    "chunk_size_chars": 42,
    "graph": {
        "nodes": [
            {"id": "a", "name": "Alice", "type": "Person", "description": "founder"},
            {"id": "b", "name": "Bob", "type": "Person", "description": "cto"},
            {"id": "acme", "name": "Acme", "type": "Organization", "description": "company"},
        ],
        "edges": [
            {"source_node_id": "a", "target_node_id": "acme", "relationship_name": "founded"}
        ],
    },
}

FUZZY_MATCH = {
    "chunk_id": "chunk-1",
    "chunk_index": 0,
    "lookups": 3,
    "count": 1,
    "matches": [
        {
            "category": "Person",
            "extracted": "Alice",
            "normalized": "alice",
            "canonical": "Alice Smith",
            "uri": "https://onto.test/Alice",
            "matched": True,
        },
        {
            "category": "Person",
            "extracted": "Bob",
            "normalized": "bob",
            "canonical": None,
            "uri": None,
            "matched": False,
        },
        {
            "category": "Organization",
            "extracted": "Acme",
            "normalized": "acme",
            "canonical": None,
            "uri": None,
            "matched": False,
        },
    ],
}

DROPPED_DUPLICATES = {
    "chunk_id": "chunk-1",
    "chunk_index": 0,
    # Graph-local node ids: the extraction prompt asks for human-readable ones.
    "dropped_node_ids": ["Albert Einstein", "Acme"],
    "count": 2,
}

RETRIEVAL_CANDIDATES = {
    "query_index": 0,
    "pool": [
        {"id": "node-1", "collection": "Entity_name", "score": 0.12},
        {"id": "node-2", "collection": "Entity_name", "score": 0.3},
        {"id": "edge-1", "collection": "EdgeType_relationship_name", "score": 0.2},
    ],
    "top_k": [{"source": "node-1", "target": "node-2", "rel": "founded", "score": 0.62}],
    "pool_size": 3,
    "pool_truncated": False,
    "cut_size": 1,
}

# Sentinels standing in for what the LLM extracted: a leak is a substring match.
EXTRACTED = ("Albert Einstein", "founded", "a physicist", "Acme", "onto.test")

# One payload per kind, shaped like its emit point builds it, every extracted
# string a sentinel. A metadata kind's sample carries none by construction —
# ids, hashes, counts, scores, model names — which is what the test checks.
SAMPLES = {
    KIND_EXTRACTION_CHUNK_GRAPH: {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "chunk_size_chars": 42,
        "graph": {
            "nodes": [
                {
                    "id": "Albert Einstein",
                    "name": "Albert Einstein",
                    "type": "Person",
                    "description": "a physicist",
                },
                {
                    "id": "Acme",
                    "name": "Acme",
                    "type": "Organization",
                    "description": "a physicist",
                },
            ],
            "edges": [
                {
                    "source_node_id": "Albert Einstein",
                    "target_node_id": "Acme",
                    "relationship_name": "founded",
                }
            ],
        },
    },
    KIND_EXTRACTION_DROPPED_DUPLICATES: DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH: {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "lookups": 1,
        "count": 1,
        "matches": [
            {
                "category": "Person",
                "extracted": "Albert Einstein",
                "normalized": "albert einstein",
                "canonical": "Albert Einstein",
                "uri": "https://onto.test/Einstein",
                "matched": True,
            }
        ],
    },
    KIND_RETRIEVAL_CANDIDATES: RETRIEVAL_CANDIDATES,
    KIND_STORAGE_DELTA: {
        "node_count": 2,
        "edge_count": 1,
        "custom_edge_count": 0,
        "node_types": {"Entity": 2},
        "node_ids": [str(uuid4()), str(uuid4())],
        "pipeline_run_id": str(uuid4()),
    },
    KIND_SUMMARY_GENERATED: {
        "chunk_id": str(uuid4()),
        "summary_id": str(uuid4()),
        "model": "openai/gpt-5-mini",
        "prompt_fingerprint": "sha256:9c1f",
        "source_text_hash": "sha256:77ab",
        "summary_chars": 120,
    },
    KIND_RUN_MANIFEST: {
        "run_id": str(uuid4()),
        "parent_run_id": None,
        "dataset_id": str(uuid4()),
        "kind": "pipeline",
        "pipeline_name": "cognify_pipeline",
        "outcome": "succeeded",
        "error_class": None,
        "sampled": True,
        "setup": {
            "llm.provider": "openai",
            "llm.model": "openai/gpt-5-mini",
            "graph_extractor": "llm",
        },
        "setup_fingerprint": "sha256:4e1d",
        "extraction.extractor": "llm",
        "extraction.model": "openai/gpt-5-mini",
        "extraction.prompt_fingerprint": "sha256:0a2c",
        "chunking.chunker": "TextChunker",
        "ontology.mode": "annotate",
        "started_at": 1.0,
        "ended_at": 2.0,
        "duration_s": 1.0,
        "counters": {"extraction.chunks": 3},
        "events_emitted": 4,
        "events_dropped": 0,
        "events_delivered": 4,
        "capture_complete": True,
    },
    KIND_SETUP_CHANGED: {
        "run_kind": "pipeline",
        "previous_fingerprint": "sha256:b9f3",
        "fingerprint": "sha256:4e1d",
        "changed": {"llm.model": {"from": "openai/gpt-5.6-luna", "to": "openai/gpt-5-mini"}},
        "setup": {"llm.model": "openai/gpt-5-mini"},
    },
}


@pytest.fixture(autouse=True)
def fixed_identity(monkeypatch):
    """No id files read or written, no LLM_API_KEY derivation: a fixed identity block."""
    monkeypatch.setattr(proxy_sink_module, "telemetry_identity", lambda: dict(IDENTITY))


class RecordingPoster:
    """The transport, recording instead of posting; optionally failing or stalling."""

    def __init__(self, fail_with: Exception | None = None, stall_s: float = 0.0):
        self.calls: list[tuple[str, dict, float]] = []
        self.fail_with = fail_with
        self.stall_s = stall_s

    def __call__(self, url: str, body: bytes, timeout_s: float) -> None:
        if self.stall_s:
            time.sleep(self.stall_s)
        self.calls.append((url, json.loads(body), timeout_s))
        if self.fail_with is not None:
            raise self.fail_with

    @property
    def records(self) -> list[dict]:
        return [record for _, body, _ in self.calls for record in body["records"]]


def _record(kind, payload, *, run_id=None, dataset_id=None, user_id=None, tenant_id=None):
    return {
        "schema_version": 1,
        "event_id": f"proc-{uuid4().hex[:8]}",
        "kind": kind,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "stage": None,
        "ts": 1.0,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# What leaves: metadata by default, content only on the switch
# ---------------------------------------------------------------------------


def test_content_kinds_leave_as_counts_and_histograms_by_default():
    graph = prepare_record(_record(KIND_EXTRACTION_CHUNK_GRAPH, CHUNK_GRAPH), include_content=False)
    assert graph["payload"] == {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "chunk_size_chars": 42,
        "node_count": 3,
        "edge_count": 1,
        "node_types": {"Person": 2, "Organization": 1},
    }
    assert graph["content_included"] is False
    assert "Alice" not in json.dumps(graph) and "founded" not in json.dumps(graph)

    fuzzy = prepare_record(_record(KIND_EXTRACTION_FUZZY_MATCH, FUZZY_MATCH), include_content=False)
    assert fuzzy["payload"] == {
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "lookups": 3,
        "count": 1,
        "categories": {
            "Person": {"lookups": 2, "matched": 1},
            "Organization": {"lookups": 1, "matched": 0},
        },
    }
    assert fuzzy["content_included"] is False
    assert "Alice" not in json.dumps(fuzzy) and "onto.test" not in json.dumps(fuzzy)


def test_dropped_duplicate_ids_and_relationship_labels_never_leave_by_default():
    dropped = prepare_record(
        _record(KIND_EXTRACTION_DROPPED_DUPLICATES, DROPPED_DUPLICATES), include_content=False
    )
    assert dropped["payload"] == {"chunk_id": "chunk-1", "chunk_index": 0, "count": 2}
    assert dropped["content_included"] is False

    candidates = prepare_record(
        _record(KIND_RETRIEVAL_CANDIDATES, RETRIEVAL_CANDIDATES), include_content=False
    )
    assert candidates["payload"] == {
        "query_index": 0,
        "pool_size": 3,
        "pool_truncated": False,
        "cut_size": 1,
        "pool": RETRIEVAL_CANDIDATES["pool"],
        "top_k": [{"source": "node-1", "target": "node-2", "score": 0.62}],
    }
    assert candidates["content_included"] is False
    assert "founded" not in json.dumps(candidates)

    # The switch sends both verbatim and flagged, like the other content kinds.
    for kind, payload in (
        (KIND_EXTRACTION_DROPPED_DUPLICATES, DROPPED_DUPLICATES),
        (KIND_RETRIEVAL_CANDIDATES, RETRIEVAL_CANDIDATES),
    ):
        full = prepare_record(_record(kind, payload), include_content=True)
        assert full["payload"] == payload
        assert full["content_included"] is True


def test_every_kind_is_classified_and_no_extracted_string_leaves_with_content_off():
    every_kind = {value for name, value in vars(events_module).items() if name.startswith("KIND_")}
    # A new kind has to say what it carries: a sample here, and a place in
    # CONTENT_KINDS when its payload holds anything the LLM extracted.
    assert set(SAMPLES) == every_kind

    for kind, payload in SAMPLES.items():
        out = prepare_record(_record(kind, payload), include_content=False)
        wire = json.dumps(out)
        assert out["content_included"] is False
        leaked = [sentinel for sentinel in EXTRACTED if sentinel in wire]
        assert not leaked, (kind, leaked)
        if kind not in CONTENT_KINDS:
            assert out["payload"] == payload


def test_a_graph_that_failed_to_dump_still_reduces_to_a_record():
    # capture_chunk_graphs records ``graph: None`` when the dump raised.
    reduced = prepare_record(
        _record(KIND_EXTRACTION_CHUNK_GRAPH, {**CHUNK_GRAPH, "graph": None}),
        include_content=False,
    )
    assert reduced["payload"]["node_count"] is None
    assert reduced["payload"]["edge_count"] is None
    assert reduced["payload"]["node_types"] is None


def test_the_content_switch_sends_the_payload_verbatim_and_says_so():
    full = prepare_record(_record(KIND_EXTRACTION_CHUNK_GRAPH, CHUNK_GRAPH), include_content=True)
    assert full["payload"] == CHUNK_GRAPH
    assert full["content_included"] is True

    # A metadata kind is never flagged as content, whichever way the switch is set.
    delta = {"node_count": 3, "edge_count": 1, "node_types": {"Person": 2}, "node_ids": ["x"]}
    for include_content in (False, True):
        out = prepare_record(_record(KIND_STORAGE_DELTA, delta), include_content=include_content)
        assert out["payload"] == delta
        assert out["content_included"] is False


def test_ids_leave_as_they_are_and_the_input_record_is_untouched():
    run_id, dataset_id, user_id, tenant_id = (str(uuid4()) for _ in range(4))
    manifest = {"run_id": run_id, "dataset_id": dataset_id, "kind": "pipeline"}
    record = _record(
        KIND_RUN_MANIFEST,
        manifest,
        run_id=run_id,
        dataset_id=dataset_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )
    before = json.dumps(record, sort_keys=True)

    out = prepare_record(record, include_content=False)

    # Ids are not content: telemetry sends dataset ids and pipeline_run_id raw too.
    assert (out["run_id"], out["dataset_id"]) == (run_id, dataset_id)
    assert (out["user_id"], out["tenant_id"]) == (user_id, tenant_id)
    assert out["payload"] == manifest
    assert "content_included" not in record
    assert json.dumps(record, sort_keys=True) == before


def test_model_settings_that_are_paths_leave_as_local_path():
    manifest = {
        "setup": {
            "llm.model": "/Users/someone/models/local.gguf",
            "embedding.model": "openai/text-embedding-3-large",
            "llm.provider": "custom",
        },
        "extraction.model": "/home/someone/checkpoints/extract.bin",
        "summarization.model": "openai/gpt-5-mini",
    }
    out = prepare_record(_record(KIND_RUN_MANIFEST, manifest), include_content=False)
    assert out["payload"]["setup"]["llm.model"] == "local_path"
    assert out["payload"]["setup"]["embedding.model"] == "openai/text-embedding-3-large"
    assert out["payload"]["setup"]["llm.provider"] == "custom"
    assert out["payload"]["extraction.model"] == "local_path"
    assert out["payload"]["summarization.model"] == "openai/gpt-5-mini"

    changed = {
        "run_kind": "pipeline",
        "changed": {
            "llm.model": {"from": "/Users/someone/models/local.gguf", "to": "openai/gpt-5-mini"},
            "llm.provider": {"from": "custom", "to": "openai"},
        },
        "setup": {"llm.model": "openai/gpt-5-mini"},
    }
    out = prepare_record(_record(KIND_SETUP_CHANGED, changed), include_content=False)
    assert out["payload"]["changed"]["llm.model"] == {
        "from": "local_path",
        "to": "openai/gpt-5-mini",
    }
    assert out["payload"]["changed"]["llm.provider"] == {"from": "custom", "to": "openai"}

    summary = {"chunk_id": "c", "model": "C:\\models\\summarizer", "summary_chars": 12}
    out = prepare_record(_record(KIND_SUMMARY_GENERATED, summary), include_content=False)
    assert out["payload"] == {"chunk_id": "c", "model": "local_path", "summary_chars": 12}


# ---------------------------------------------------------------------------
# The request: identity block, prepared records, byte caps, off-loop transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_bodies_carry_the_identity_block_and_the_prepared_records():
    poster = RecordingPoster()
    sink = ProxySink("https://proxy.test/capture", poster=poster, timeout_s=2.5)
    records = [
        _record(KIND_EXTRACTION_CHUNK_GRAPH, CHUNK_GRAPH, run_id="r1"),
        _record(KIND_STORAGE_DELTA, {"node_count": 3}, run_id="r1"),
    ]

    await sink(records)

    [(url, body, timeout_s)] = poster.calls
    assert url == "https://proxy.test/capture"
    assert timeout_s == 2.5
    assert body["identity"] == IDENTITY
    assert [record["kind"] for record in body["records"]] == [
        KIND_EXTRACTION_CHUNK_GRAPH,
        KIND_STORAGE_DELTA,
    ]
    assert "graph" not in body["records"][0]["payload"]  # reduced
    assert body["records"][0]["event_id"] == records[0]["event_id"]
    assert all("content_included" not in record for record in records)  # not mutated


@pytest.mark.asyncio
async def test_batches_split_by_bytes_and_oversize_records_are_skipped():
    poster = RecordingPoster()
    small = _record(KIND_STORAGE_DELTA, {"node_count": 1})
    one_record_bytes = len(json.dumps(prepare_record(small, include_content=False)))
    frame = len(b'{"identity":' + json.dumps(IDENTITY).encode() + b',"records":[]}')
    sink = ProxySink(
        "https://proxy.test/capture",
        poster=poster,
        # Room for two records per request, not three.
        max_batch_bytes=frame + 2 * one_record_bytes + 1,
        max_record_bytes=one_record_bytes + 50,
    )
    oversize = _record(KIND_STORAGE_DELTA, {"node_ids": ["x" * 400]})
    records = [small, _record(KIND_STORAGE_DELTA, {"node_count": 2}), oversize, small, small]

    await sink(records)

    assert [len(body["records"]) for _, body, _ in poster.calls] == [2, 2]
    assert oversize["event_id"] not in {record["event_id"] for record in poster.records}


@pytest.mark.asyncio
async def test_a_failed_post_counts_the_batch_as_dropped_on_the_run():
    poster = RecordingPoster(fail_with=RuntimeError("503 from the proxy"))
    capture.register_capture_sink(ProxySink("https://proxy.test/capture", poster=poster))
    hook._configure(flush_interval_s=60.0)

    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        capture.emit(KIND_SUMMARY_GENERATED, "lost", payload_kind="text")
        await capture.drain()

    assert len(poster.calls) == 1
    assert scope.emitted == 1
    assert scope.dropped == 1
    assert scope.delivered == 0


@pytest.mark.asyncio
async def test_the_post_runs_off_the_event_loop():
    """A slow proxy must not freeze the flusher's loop (or the recall awaiting on it)."""
    poster = RecordingPoster(stall_s=0.2)
    sink = ProxySink("https://proxy.test/capture", poster=poster)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    try:
        await sink([_record(KIND_STORAGE_DELTA, {"node_count": 1})])
    finally:
        task.cancel()

    assert ticks >= 5
    assert len(poster.calls) == 1


# ---------------------------------------------------------------------------
# Selection and kill switches
# ---------------------------------------------------------------------------


def test_default_url_is_the_telemetry_proxys_capture_route(tmp_path):
    from cognee.shared import utils

    assert default_proxy_url() == utils.proxy_url.rstrip("/") + "/capture"
    sink = ProxySink.from_config(
        CaptureConfig(cognee_capture_sink="proxy", cognee_capture_dir=str(tmp_path))
    )
    assert sink.url == default_proxy_url()
    assert sink.include_content is False
    assert sink.timeout_s == 5.0


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("TELEMETRY_DISABLED", "1", "TELEMETRY_DISABLED is set"),
        ("ENV", "test", "ENV=test"),
        ("ENV", "dev", "ENV=dev"),
    ],
)
def test_the_telemetry_kill_switches_block_the_proxy_sink(monkeypatch, name, value, reason):
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK", "proxy")

    assert proxy_sink_blocked() == reason
    # proxy alone: nothing to deliver to, capture stays off.
    assert capture.is_active() is False
    assert hook._runtime.sink is None


def test_both_degrades_to_storage_when_the_proxy_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK", "both")
    monkeypatch.setenv("COGNEE_CAPTURE_DIR", str(tmp_path))

    assert capture.is_active() is True
    assert isinstance(hook._runtime.sink, StorageSink)


def test_the_environment_selects_the_proxy_sink(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK", "proxy")
    monkeypatch.setenv("COGNEE_CAPTURE_PROXY_URL", "https://proxy.test/capture")
    monkeypatch.setenv("COGNEE_CAPTURE_PROXY_CONTENT", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_PROXY_TIMEOUT_S", "1.5")
    monkeypatch.setenv("COGNEE_CAPTURE_DIR", str(tmp_path))

    assert capture.is_active() is True
    sink = hook._runtime.sink
    assert isinstance(sink, ProxySink)
    assert sink.url == "https://proxy.test/capture"
    assert sink.include_content is True
    assert sink.timeout_s == 1.5


def test_both_fans_out_to_storage_and_proxy(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_SINK", "both")
    monkeypatch.setenv("COGNEE_CAPTURE_DIR", str(tmp_path))

    assert capture.is_active() is True
    sink = hook._runtime.sink
    assert isinstance(sink, FanOutSink)
    assert [type(inner) for inner in sink.sinks] == [StorageSink, ProxySink]


@pytest.mark.asyncio
async def test_fan_out_runs_every_sink_and_reraises_the_first_failure():
    first, last = FakeCaptureSink(), FakeCaptureSink()

    async def failing(records):
        raise RuntimeError("proxy down")

    sink = FanOutSink([first, failing, last])
    records = [_record(KIND_STORAGE_DELTA, {"node_count": 1})]

    with pytest.raises(RuntimeError, match="proxy down"):
        await sink(records)

    assert first.records == records
    assert last.records == records
    with pytest.raises(ValueError, match="at least one sink"):
        FanOutSink([])

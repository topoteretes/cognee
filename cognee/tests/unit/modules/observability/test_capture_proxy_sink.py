"""ProxySink (SDK-529): what leaves the process on the way to the analytics proxy —
metadata by default, the identity telemetry sends, masked model paths — in
byte-capped batches, off the event loop, and only when the telemetry kill
switches allow it. No network: the poster is a recorder."""

import asyncio
import json
import os
import stat
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
from cognee.modules.observability.capture.sinks import PartialDelivery
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


TEST_ID_KEY = b"test-id-key"


@pytest.fixture(autouse=True)
def fixed_id_key(monkeypatch):
    """One known pseudonym key, no key file read or written — and the kill
    switches off, so a sink called directly is not blocked by the shell's ENV."""
    monkeypatch.setenv(proxy_sink_module.ID_KEY_ENV, TEST_ID_KEY.decode())
    monkeypatch.delenv("TELEMETRY_DISABLED", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    proxy_sink_module._reset_id_key_for_tests()
    yield
    proxy_sink_module._reset_id_key_for_tests()


def _pseudonym(value):
    return proxy_sink_module.graph_id_pseudonym(value, TEST_ID_KEY)


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
        "chunk_id": _pseudonym("chunk-1"),
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
        "chunk_id": _pseudonym("chunk-1"),
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
    assert dropped["payload"] == {"chunk_id": _pseudonym("chunk-1"), "chunk_index": 0, "count": 2}
    assert dropped["content_included"] is False

    candidates = prepare_record(
        _record(KIND_RETRIEVAL_CANDIDATES, RETRIEVAL_CANDIDATES), include_content=False
    )
    assert candidates["payload"] == {
        "query_index": 0,
        "pool_size": 3,
        "pool_truncated": False,
        "cut_size": 1,
        "pool": [
            {**entry, "id": _pseudonym(entry["id"])} for entry in RETRIEVAL_CANDIDATES["pool"]
        ],
        "top_k": [{"source": _pseudonym("node-1"), "target": _pseudonym("node-2"), "score": 0.62}],
    }
    assert candidates["content_included"] is False
    assert "founded" not in json.dumps(candidates)

    # The switch sends both verbatim and flagged, like the other content kinds —
    # verbatim but for the graph ids, which only ever leave as pseudonyms.
    for kind, payload in (
        (KIND_EXTRACTION_DROPPED_DUPLICATES, DROPPED_DUPLICATES),
        (KIND_RETRIEVAL_CANDIDATES, RETRIEVAL_CANDIDATES),
    ):
        full = prepare_record(_record(kind, payload), include_content=True)
        assert full["payload"] == proxy_sink_module.pseudonymize_graph_ids(
            kind, payload, TEST_ID_KEY
        )
        assert full["content_included"] is True
    assert full["payload"]["top_k"][0]["rel"] == "founded"
    assert full["payload"]["top_k"][0]["source"] == _pseudonym("node-1")


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
            # As it is, but for the graph ids, which leave as pseudonyms.
            assert out["payload"] == proxy_sink_module.pseudonymize_graph_ids(
                kind, payload, TEST_ID_KEY
            )


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
    assert full["payload"] == {**CHUNK_GRAPH, "chunk_id": _pseudonym("chunk-1")}
    assert full["content_included"] is True

    # A metadata kind is never flagged as content, whichever way the switch is set.
    delta = {"node_count": 3, "edge_count": 1, "node_types": {"Person": 2}, "node_ids": ["x"]}
    for include_content in (False, True):
        out = prepare_record(_record(KIND_STORAGE_DELTA, delta), include_content=include_content)
        assert out["payload"] == {**delta, "node_ids": [_pseudonym("x")]}
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


def test_graph_ids_leave_as_pseudonyms_under_a_key_that_never_leaves():
    """An entity id is uuid5(public namespace, "Entity:<name>"): anyone holding
    a candidate name can compute it and confirm it appears in a record. What
    leaves is a keyed hash instead — the same for the same id across records
    (joins hold), different under another installation's key — and the key
    itself is in no request."""
    node_id, chunk_id = str(uuid4()), str(uuid4())
    delta = {"node_count": 1, "node_types": {"Entity": 1}, "node_ids": [node_id, None]}
    candidates = {
        **RETRIEVAL_CANDIDATES,
        "pool": [{"id": node_id, "collection": "Entity_name", "score": 0.1}],
        "top_k": [{"source": node_id, "target": chunk_id, "rel": "mentions", "score": 0.5}],
    }
    summary = {"chunk_id": chunk_id, "summary_id": str(uuid4()), "model": "m", "summary_chars": 3}

    out_delta = prepare_record(_record(KIND_STORAGE_DELTA, delta), include_content=False)
    out_candidates = prepare_record(
        _record(KIND_RETRIEVAL_CANDIDATES, candidates), include_content=False
    )
    out_summary = prepare_record(_record(KIND_SUMMARY_GENERATED, summary), include_content=False)

    pseudonym = out_delta["payload"]["node_ids"][0]
    assert pseudonym != node_id and len(pseudonym) == proxy_sink_module.PSEUDONYM_HEX_CHARS
    assert out_delta["payload"]["node_ids"][1] is None
    # The same node, the same pseudonym, whichever record it appears in.
    assert out_candidates["payload"]["pool"][0]["id"] == pseudonym
    assert out_candidates["payload"]["top_k"][0]["source"] == pseudonym
    assert out_candidates["payload"]["top_k"][0]["target"] == out_summary["payload"]["chunk_id"]
    assert out_summary["payload"]["summary_id"] != summary["summary_id"]
    wire = json.dumps([out_delta, out_candidates, out_summary])
    assert node_id not in wire and chunk_id not in wire and TEST_ID_KEY.decode() not in wire
    # Content mode changes nothing about the ids.
    with_content = prepare_record(
        _record(KIND_RETRIEVAL_CANDIDATES, candidates), include_content=True
    )
    assert with_content["payload"]["pool"][0]["id"] == pseudonym
    # Another installation's key is another pseudonym space.
    other = prepare_record(
        _record(KIND_STORAGE_DELTA, delta), include_content=False, id_key=b"other"
    )
    assert other["payload"]["node_ids"][0] not in (pseudonym, node_id)
    # A UUID and its string form agree, so an emit point may pass either.
    as_uuid = uuid4()
    assert proxy_sink_module.graph_id_pseudonym(as_uuid, TEST_ID_KEY) == _pseudonym(str(as_uuid))


def test_the_installation_key_is_created_once_owner_only_and_read_back(monkeypatch, tmp_path):
    from cognee.shared import utils as shared_utils

    monkeypatch.delenv(proxy_sink_module.ID_KEY_ENV, raising=False)
    monkeypatch.setattr(shared_utils, "_PERSISTENT_ID_DIR", tmp_path / ".cognee")
    proxy_sink_module._reset_id_key_for_tests()

    first = proxy_sink_module.installation_id_key()
    key_file = tmp_path / ".cognee" / proxy_sink_module.ID_KEY_FILE_NAME
    assert key_file.read_text(encoding="utf-8").encode("utf-8") == first
    assert len(first) == 64  # 32 random bytes, as hex
    if os.name != "nt":
        assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

    proxy_sink_module._reset_id_key_for_tests()  # a later process
    assert proxy_sink_module.installation_id_key() == first

    # The environment wins: one pseudonym space across machines. The file stays.
    monkeypatch.setenv(proxy_sink_module.ID_KEY_ENV, "shared")
    proxy_sink_module._reset_id_key_for_tests()
    assert proxy_sink_module.installation_id_key() == b"shared"
    assert key_file.read_text(encoding="utf-8").encode("utf-8") == first


def test_an_unwritable_home_leaves_a_key_for_this_process_only(monkeypatch, tmp_path):
    from cognee.shared import utils as shared_utils

    monkeypatch.delenv(proxy_sink_module.ID_KEY_ENV, raising=False)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the directory should be", encoding="utf-8")
    monkeypatch.setattr(shared_utils, "_PERSISTENT_ID_DIR", blocked)
    proxy_sink_module._reset_id_key_for_tests()

    key = proxy_sink_module.installation_id_key()
    assert len(key) == 32 and proxy_sink_module.installation_id_key() == key
    out = prepare_record(_record(KIND_STORAGE_DELTA, {"node_ids": ["n"]}), include_content=False)
    assert out["payload"]["node_ids"] == [proxy_sink_module.graph_id_pseudonym("n", key)]
    assert "n" != out["payload"]["node_ids"][0]


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
    assert out["payload"] == {
        "chunk_id": _pseudonym("c"),
        "model": "local_path",
        "summary_chars": 12,
    }


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


def _two_records_per_request(poster, **overrides):
    small = _record(KIND_STORAGE_DELTA, {"node_count": 1})
    one_record_bytes = len(json.dumps(prepare_record(small, include_content=False)))
    frame = len(b'{"identity":' + json.dumps(IDENTITY).encode() + b',"records":[]}')
    options = {
        # Room for two records per request, not three.
        "max_batch_bytes": frame + 2 * one_record_bytes + 1,
        "max_record_bytes": one_record_bytes + 50,
    }
    options.update(overrides)
    return ProxySink("https://proxy.test/capture", poster=poster, **options)


@pytest.mark.asyncio
async def test_batches_split_by_bytes_and_an_oversize_record_is_left_out_and_reported():
    poster = RecordingPoster()
    sink = _two_records_per_request(poster)
    small = _record(KIND_STORAGE_DELTA, {"node_count": 1})
    # Padded through a field that leaves as it is (node ids would shrink to pseudonyms).
    oversize = _record(KIND_STORAGE_DELTA, {"node_count": 1, "node_types": {"x" * 400: 1}})
    records = [small, _record(KIND_STORAGE_DELTA, {"node_count": 2}), oversize, small, small]

    with pytest.raises(PartialDelivery) as info:
        await sink(records)

    assert info.value.undelivered == (2,)  # reported by position, after the POSTs
    assert [len(body["records"]) for _, body, _ in poster.calls] == [2, 2]
    assert oversize["event_id"] not in {record["event_id"] for record in poster.records}


@pytest.mark.asyncio
async def test_a_record_the_sink_leaves_out_is_charged_to_its_run_not_credited():
    """Over the record cap, or not serializable: left out of the request and
    dropped on the run — the POST succeeding must not credit it as delivered."""
    poster = RecordingPoster()
    capture.register_capture_sink(
        ProxySink("https://proxy.test/capture", poster=poster, max_record_bytes=600)
    )
    hook._configure(flush_interval_s=60.0)
    circular: dict = {}
    circular["self"] = circular

    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        capture.emit(KIND_STORAGE_DELTA, {"node_count": 1}, payload_kind="json")
        # Padded through a field that leaves as it is (node ids would shrink to pseudonyms).
        capture.emit(KIND_STORAGE_DELTA, {"node_types": {"x" * 2000: 1}}, payload_kind="json")
        capture.emit(KIND_STORAGE_DELTA, circular, payload_kind="json")
        await capture.drain()
        assert (scope.emitted, scope.dropped, scope.delivered) == (3, 2, 1)

    assert len(poster.calls) == 1
    assert [record["payload"] for record in poster.records] == [{"node_count": 1}]


@pytest.mark.asyncio
async def test_a_post_that_fails_after_an_earlier_one_landed_charges_only_what_it_carried():
    class FailsSecond(RecordingPoster):
        def __call__(self, url, body, timeout_s):
            super().__call__(url, body, timeout_s)
            if len(self.calls) == 2:
                raise RuntimeError("503 from the proxy")

    poster = FailsSecond()
    sink = _two_records_per_request(poster)
    records = [_record(KIND_STORAGE_DELTA, {"node_count": 1}) for _ in range(5)]

    with pytest.raises(PartialDelivery) as info:
        await sink(records)

    # The first request landed; the failed one and the one never sent did not.
    assert info.value.undelivered == (2, 3, 4)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert len(poster.calls) == 2
    assert [r["event_id"] for r in poster.records[:2]] == [r["event_id"] for r in records[:2]]


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
async def test_the_kill_switch_is_honoured_on_every_send_not_only_at_selection(monkeypatch):
    """TELEMETRY_DISABLED is the one switch that promises nothing leaves. A sink
    selected while it was unset refuses the moment it is set, and the records
    it refused are dropped on their run, not credited."""
    poster = RecordingPoster()
    capture.register_capture_sink(ProxySink("https://proxy.test/capture", poster=poster))
    hook._configure(flush_interval_s=60.0)

    with capture.run_scope(uuid4(), kind="pipeline") as scope:
        capture.emit(KIND_SUMMARY_GENERATED, "before", payload_kind="text")
        await capture.drain()
        monkeypatch.setenv("TELEMETRY_DISABLED", "1")
        capture.emit(KIND_SUMMARY_GENERATED, "after", payload_kind="text")
        await capture.drain()

    assert [record["payload"] for record in poster.records] == ["before"]
    assert (scope.emitted, scope.dropped, scope.delivered) == (2, 1, 1)


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


@pytest.mark.asyncio
async def test_fan_out_unions_partial_deliveries_and_a_failure_still_fails_the_batch():
    def leaves_out(*positions):
        async def sink(records):
            raise PartialDelivery(positions)

        return sink

    stored = FakeCaptureSink()
    records = [_record(KIND_STORAGE_DELTA, {"node_count": n}) for n in range(3)]

    with pytest.raises(PartialDelivery) as info:
        await FanOutSink([leaves_out(0), stored, leaves_out(2)])(records)
    # "Did the warehouse get it", per record: what any sink left out is lost.
    assert info.value.undelivered == (0, 2)
    assert stored.records == records

    async def failing(records):
        raise RuntimeError("proxy down")

    with pytest.raises(RuntimeError, match="proxy down"):
        await FanOutSink([leaves_out(0), failing])(records)

"""Eval capture end to end (SDK-529): ONE cognify-shaped pipeline run through the
real runner, followed by ONE recorded search, observed through a FakeCaptureSink —
and once more with capture switched on through the environment, so the real
``StorageSink`` persists the same run in its documented on-disk layout.

What is real: ``run_tasks`` (via ``runner_plumbing``), the per-item task resolver
notes from ``cognify``, ``classify_documents``, ``extract_chunks_from_documents`` with
the real ``TextChunker`` over a real temp file, ``extract_graph_and_summarize`` (the
real ``extract_content_graph`` prompt rendering and ``extract_summary_with_provenance``
prompt read, the dedup, the ontology canonicalization, ``construct_data_points_*``),
``add_data_points`` (real ``get_graph_from_model`` + dedup over the produced
DataPoints), ``record_operation`` and ``brute_force_triplet_search``.

What is faked: the LLM (``LLMGateway.acreate_structured_output``), the embedding
tokenizer, every database touch (relational session, graph/vector engines, edge
lookup, operation-row writer), and the retrieval vector engine / memory fragment —
the same fakes the per-stage unit tests use.
"""

import gzip
import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

import pytest

import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)
from cognee.infrastructure.llm.config import get_llm_config, get_llm_context_config
from cognee.infrastructure.llm.extraction.extract_summary import SUMMARY_PROMPT_FILE
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RETRIEVAL_CANDIDATES,
    KIND_RUN_MANIFEST,
    KIND_STORAGE_DELTA,
    KIND_SUMMARY_GENERATED,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node as KGNode, SummarizedContent
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.storage.add_data_points import add_data_points

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
record_operation_module = importlib.import_module("cognee.modules.operations.record_operation")
egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")
adp_module = importlib.import_module("cognee.tasks.storage.add_data_points")
chunks_module = importlib.import_module("cognee.tasks.documents.extract_chunks_from_documents")
sentence_module = importlib.import_module("cognee.tasks.chunks.chunk_by_sentence")

pytestmark = pytest.mark.usefixtures("capture_reset")

MAX_CHUNK_SIZE = 12  # words (the fake tokenizer counts one token per word)
PARAGRAPH_ONE = "Alice met Bob in Paris. They founded Acme Corp together.\n\n"
PARAGRAPH_TWO = "Carol joined Acme Corp later. She leads the research team in Berlin.\n"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Canonicalizer(BaseOntologyResolver):
    """Resolves the ``Person`` class to an ontology node; everything else misses."""

    def __init__(self):
        super().__init__(FuzzyMatchingStrategy(cutoff=0.85))

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "person":
            root = AttachedOntologyNode("https://example.test/onto#Person", "classes")
            return [root], [], root
        return [], [], None


def _extracted_graph(chunk_text: str) -> KnowledgeGraph:
    """A per-chunk graph with one duplicated node id, keyed on the chunk text."""
    tag = str(uuid5(NAMESPACE_OID, chunk_text))[:8]
    return KnowledgeGraph(
        nodes=[
            KGNode(id=f"p-{tag}", name=f"Person {tag}", type="Person", description="first"),
            KGNode(id=f"p-{tag}", name=f"Duplicate {tag}", type="Person", description="dupe"),
            KGNode(id=f"o-{tag}", name=f"Org {tag}", type="Organization", description="org"),
        ],
        edges=[
            KGEdge(
                source_node_id=f"p-{tag}", target_node_id=f"o-{tag}", relationship_name="works_at"
            )
        ],
    )


async def _fake_structured_output(text_input, system_prompt, response_model, **kwargs):
    if response_model is KnowledgeGraph:
        return _extracted_graph(text_input)
    if response_model is SummarizedContent:
        return SummarizedContent(summary=f"Summary of: {text_input[:20]}", description="d")
    raise AssertionError(f"unexpected response model {response_model!r}")


def _graph_provenance_unified():
    """A unified engine whose graph is marked graph-provenance (ledger skipped)."""
    graph_engine = AsyncMock()
    graph_engine.is_empty = AsyncMock(return_value=True)
    graph_engine.get_graph_metadata = AsyncMock(
        return_value={
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
        }
    )
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = AsyncMock()
    unified.has_capability = lambda capability: False
    return unified, graph_engine


class _ScoredResult:
    def __init__(self, id, score):
        self.id = id
        self.score = score
        self.payload = {}


def _retrieval_edge(source, target, rel, distances):
    node1 = Node(source, {"name": source})
    node2 = Node(target, {"name": target})
    edge = Edge(node1, node2, attributes={"relationship_type": rel})
    for element, element_distances in zip((node1, edge, node2), distances):
        element.attributes["vector_distance"] = list(element_distances)
    return edge


def _by_kind(sink, kind):
    return [record for record in sink.records if record["kind"] == kind]


def _assert_flat(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_flat(item)
    elif isinstance(value, list):
        for item in value:
            _assert_flat(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), repr(value)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_fakes(monkeypatch, tmp_path, runner_plumbing):
    """Stub every database and LLM touch a cognify-shaped run makes; return the collaborators."""
    dataset = SimpleNamespace(id=uuid4(), name="probe_dataset", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_progress", AsyncMock())
    # The source document is read through open_data_file's local allowlist.
    monkeypatch.setenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", str(tmp_path))

    # LLM: one dispatcher for extraction (KnowledgeGraph) and summarization.
    llm = AsyncMock(side_effect=_fake_structured_output)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", llm)

    # Chunking: no embedding engine / tokenizer, no relational token-count write.
    monkeypatch.setattr(
        sentence_module, "get_embedding_engine", lambda: SimpleNamespace(tokenizer=None)
    )
    monkeypatch.setattr(chunks_module, "update_document_token_count", AsyncMock())

    # Extraction: no graph lookup for existing edges.
    monkeypatch.setattr(egd_module, "find_existing_edge_identities", AsyncMock(return_value=set()))

    # Storage: graph-provenance-marked unified engine, no vector indexing.
    unified, graph_engine = _graph_provenance_unified()
    monkeypatch.setattr(adp_module, "get_unified_engine", AsyncMock(return_value=unified))
    monkeypatch.setattr(adp_module, "index_data_points", AsyncMock())
    monkeypatch.setattr(adp_module, "index_graph_edges", AsyncMock())

    # record_operation: no pipeline_runs row.
    monkeypatch.setattr(record_operation_module, "_write_operation_row", AsyncMock())

    return SimpleNamespace(dataset=dataset, llm=llm, graph_engine=graph_engine)


def _cognify_tasks(resolver: BaseOntologyResolver) -> list[Task]:
    """The standard cognify task list shape (get_default_tasks), with test knobs."""
    return [
        Task(classify_documents),
        Task(extract_chunks_from_documents, max_chunk_size=MAX_CHUNK_SIZE, chunker=TextChunker),
        Task(
            extract_graph_and_summarize,
            graph_model=KnowledgeGraph,
            config={
                "ontology_config": {"ontology_resolver": resolver, "ontology_mode": "annotate"}
            },
            custom_prompt=None,
            task_config={"batch_size": 2000},
        ),
        Task(add_data_points, embed_triplets=False, task_config={"batch_size": 2000}),
    ]


async def _run_pipeline(tmp_path, fakes, user):
    source = tmp_path / "notes.txt"
    source.write_text(PARAGRAPH_ONE + PARAGRAPH_TWO, encoding="utf-8")
    data_item = SimpleNamespace(
        id=uuid4(),
        name="notes",
        extension="txt",
        raw_data_location=str(source),
        mime_type="text/plain",
        external_metadata={},
        importance_weight=None,
        system_metadata={},
    )

    tasks = _cognify_tasks(_Canonicalizer())

    def resolve_cognify_tasks(item):
        # Mirrors cognify()'s per-item resolver: the run scope exists only here.
        cognify_module._note_chunking_config(tasks)
        return tasks

    events = []
    async for event in run_tasks_module.run_tasks(
        tasks=resolve_cognify_tasks,
        dataset_id=fakes.dataset.id,
        data=[data_item],
        user=user,
        pipeline_name="cognify_pipeline",
    ):
        events.append(event)
    return events


async def _run_search(dataset_id):
    edge = _retrieval_edge("alice", "acme", "works_at", ([0.1], [0.2], [0.3]))
    engine = AsyncMock()
    engine.embedding_engine = AsyncMock()
    engine.embedding_engine.embed_text = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    pool = {
        "Entity_name": [_ScoredResult("alice", 0.1), _ScoredResult("acme", 0.4)],
        "EdgeType_relationship_name": [_ScoredResult("works_at", 0.2)],
    }
    engine.search = AsyncMock(side_effect=lambda **kw: pool.get(kw["collection_name"], []))
    fragment = AsyncMock(
        map_vector_distances_to_graph_nodes=AsyncMock(),
        map_vector_distances_to_graph_edges=AsyncMock(),
        calculate_top_triplet_importances=AsyncMock(return_value=[edge]),
    )

    with (
        patch(
            "cognee.modules.retrieval.utils.node_edge_vector_search.get_vector_engine_async",
            return_value=engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
            return_value=fragment,
        ),
    ):
        async with record_operation_module.record_operation("search") as context:
            context.set_dataset(dataset_id)
            results = await brute_force_triplet_search(
                query="who works at acme", top_k=3, collections=["Entity_name"]
            )
    return context, results


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_pipeline_run_and_one_search_yield_every_capture_kind(
    tmp_path, pipeline_fakes, fake_capture_sink
):
    fakes = pipeline_fakes
    user = SimpleNamespace(id=uuid4(), tenant_id=None, email="probe@example.test")

    async with record_operation_module.record_operation("remember") as remember_context:
        remember_context.set_dataset(fakes.dataset.id)
        events = await _run_pipeline(tmp_path, fakes, user)

    pipeline_run_id = str(events[0].pipeline_run_id)
    assert type(events[-1]).__name__ == "PipelineRunCompleted", events[-1]

    search_context, search_results = await _run_search(fakes.dataset.id)
    assert len(search_results) == 1

    # The pipeline drained its own events; the operation manifests need one drain.
    await capture.drain()
    sink = fake_capture_sink

    # Everything the sink holds is JSON: no Node/Edge/ScoredResult/DataPoint leaked.
    json.dumps(sink.records)
    for record in sink.records:
        _assert_flat(record["payload"])

    # ---- manifests: remember (operation) > pipeline > search (operation) ------
    manifests = {
        m["payload"]["kind"] + ":" + m["payload"].get("operation", ""): m
        for m in _by_kind(sink, KIND_RUN_MANIFEST)
    }
    assert set(manifests) == {"operation:remember", "pipeline:", "operation:search"}

    remember = manifests["operation:remember"]["payload"]
    assert remember["run_id"] == str(remember_context.operation_id)
    assert remember["dataset_id"] == str(fakes.dataset.id)
    assert remember["outcome"] == "succeeded"
    assert remember["error_class"] is None
    # Notes and counters land on the innermost scope (the pipeline), not the
    # enclosing operation: the two manifests are joined offline via parent_run_id.
    assert remember["counters"] == {}
    assert "extraction.model" not in remember
    # 3 chunks x (extraction + summarization).
    assert fakes.llm.await_count == 6

    pipeline = manifests["pipeline:"]["payload"]
    assert pipeline["run_id"] == pipeline_run_id
    assert pipeline["parent_run_id"] == str(remember_context.operation_id)
    assert pipeline["dataset_id"] == str(fakes.dataset.id)
    assert pipeline["sampled"] is True
    # chunking.* from the per-item resolver
    assert pipeline["chunking.chunker"] == "TextChunker"
    assert pipeline["chunking.chunk_size"] == MAX_CHUNK_SIZE
    # ontology.* from integrate_chunk_graphs
    assert pipeline["ontology.mode"] == "annotate"
    assert pipeline["ontology.resolver"] == "_Canonicalizer"
    assert pipeline["ontology.matching_strategy"] == "FuzzyMatchingStrategy"
    assert pipeline["ontology.threshold"] == 0.85
    # extraction.* from the real extract_content_graph prompt rendering
    rendered_prompt = render_prompt(get_llm_config().graph_prompt_path, {})
    assert (
        pipeline["extraction.model"]
        == get_llm_context_config().stage_config("extraction").llm_model
    )
    assert pipeline["extraction.prompt_fingerprint"] == capture.prompt_fingerprint(rendered_prompt)
    # summarization.* from the real extract_summary_with_provenance prompt read
    assert (
        pipeline["summarization.model"]
        == get_llm_context_config().stage_config("summarization").llm_model
    )
    assert pipeline["summarization.prompt_fingerprint"] == capture.prompt_fingerprint(
        read_query_prompt(SUMMARY_PROMPT_FILE)
    )

    # ---- extraction.chunk_graph: one per real chunk, pre-dedup snapshot ---------
    chunk_graphs = _by_kind(sink, KIND_EXTRACTION_CHUNK_GRAPH)
    assert len(chunk_graphs) >= 2, "the text must have been split into at least two chunks"
    assert {e["run_id"] for e in chunk_graphs} == {pipeline_run_id}
    assert {e["dataset_id"] for e in chunk_graphs} == {str(fakes.dataset.id)}
    assert {e["stage"] for e in chunk_graphs} == {"extract_graph_from_data"}
    assert [e["payload"]["chunk_index"] for e in chunk_graphs] == list(range(len(chunk_graphs)))
    chunk_ids = [e["payload"]["chunk_id"] for e in chunk_graphs]
    for event in chunk_graphs:
        graph = event["payload"]["graph"]
        assert len(graph["nodes"]) == 3, "snapshot taken before dedup"
        assert graph["nodes"][0]["type"] == "Person", "snapshot taken before canonicalization"
        assert event["payload"]["chunk_size_chars"] > 0

    # ---- extraction.dropped_duplicates: one per chunk graph that dropped ------
    dropped = _by_kind(sink, KIND_EXTRACTION_DROPPED_DUPLICATES)
    assert len(dropped) == len(chunk_graphs)
    assert {e["run_id"] for e in dropped} == {pipeline_run_id}
    for event, graph_event in zip(dropped, chunk_graphs):
        expected_id = graph_event["payload"]["graph"]["nodes"][0]["id"]
        assert event["payload"] == {
            "chunk_id": graph_event["payload"]["chunk_id"],
            "chunk_index": graph_event["payload"]["chunk_index"],
            "dropped_node_ids": [expected_id],
            "count": 1,
        }
    assert pipeline["counters"]["extraction.dropped_duplicate_nodes"] == len(chunk_graphs)

    # ---- extraction.fuzzy_match: one per chunk that triggered lookups ---------
    fuzzy = _by_kind(sink, KIND_EXTRACTION_FUZZY_MATCH)
    assert fuzzy, "the ontology resolver was consulted"
    assert {e["run_id"] for e in fuzzy} == {pipeline_run_id}
    first_matches = fuzzy[0]["payload"]["matches"]
    person = next(
        m for m in first_matches if m["category"] == "classes" and m["normalized"] == "person"
    )
    assert person["matched"] is True
    assert person["uri"] == "https://example.test/onto#Person"
    assert fuzzy[0]["payload"]["chunk_id"] == chunk_ids[0]
    assert pipeline["counters"]["extraction.fuzzy_lookups"] == sum(
        e["payload"]["lookups"] for e in fuzzy
    )
    assert pipeline["counters"]["extraction.fuzzy_matches"] == sum(
        e["payload"]["count"] for e in fuzzy
    )

    # ---- summary.generated: one per chunk, joined to the chunk and summary ids -
    summaries = _by_kind(sink, KIND_SUMMARY_GENERATED)
    assert [e["payload"]["chunk_id"] for e in summaries] == chunk_ids
    assert {e["run_id"] for e in summaries} == {pipeline_run_id}
    assert {e["stage"] for e in summaries} == {"summarize_text"}
    for event in summaries:
        payload = event["payload"]
        assert payload["summary_id"] == str(uuid5(UUID(payload["chunk_id"]), "TextSummary"))
        assert payload["model"] == pipeline["summarization.model"]
        assert payload["prompt_fingerprint"] == pipeline["summarization.prompt_fingerprint"]
        assert payload["source_text_hash"].startswith("sha256:")
        assert payload["summary_chars"] > 0
        assert "text" not in payload and "summary" not in payload

    # ---- storage.delta: one per add_data_points call, ids/types/counts only ---
    [delta] = _by_kind(sink, KIND_STORAGE_DELTA)
    assert delta["run_id"] == pipeline_run_id
    assert delta["dataset_id"] == str(fakes.dataset.id)
    assert delta["stage"] == "add_data_points"
    payload = delta["payload"]
    assert payload["pipeline_run_id"] == pipeline_run_id
    fakes.graph_engine.add_nodes.assert_awaited_once()
    written_nodes = fakes.graph_engine.add_nodes.await_args.args[0]
    assert payload["node_count"] == len(written_nodes)
    assert sorted(payload["node_ids"]) == sorted(str(node.id) for node in written_nodes)
    assert payload["node_types"]["TextSummary"] == len(chunk_ids)
    assert payload["node_types"]["DocumentChunk"] == len(chunk_ids)
    assert payload["node_types"]["TextDocument"] == 1
    assert payload["node_types"]["Entity"] == 2 * len(chunk_ids)  # person + org per chunk
    assert set(s["payload"]["summary_id"] for s in summaries) <= set(payload["node_ids"])
    assert set(chunk_ids) <= set(payload["node_ids"])
    fakes.graph_engine.add_edges.assert_awaited_once()
    assert payload["edge_count"] == len(fakes.graph_engine.add_edges.await_args.args[0])
    assert payload["custom_edge_count"] == 0
    assert pipeline["counters"]["storage.nodes_written"] == payload["node_count"]
    assert pipeline["counters"]["storage.edges_written"] == payload["edge_count"]

    # ---- retrieval.candidates: under the search operation, dataset bound late -
    search = manifests["operation:search"]["payload"]
    assert search["run_id"] == str(search_context.operation_id)
    assert search["dataset_id"] == str(fakes.dataset.id)
    assert search["outcome"] == "succeeded"
    assert search["retrieval.top_k"] == 3
    assert search["retrieval.mode"] == "single"
    assert search["retrieval.collections"] == ["Entity_name", "EdgeType_relationship_name"]

    [candidates] = _by_kind(sink, KIND_RETRIEVAL_CANDIDATES)
    assert candidates["run_id"] == str(search_context.operation_id)
    assert candidates["dataset_id"] == str(fakes.dataset.id)
    assert candidates["stage"] == "brute_force_triplet_search"
    payload = candidates["payload"]
    assert payload["query_index"] == 0
    assert payload["pool"] == [
        {"id": "alice", "collection": "Entity_name", "score": 0.1},
        {"id": "acme", "collection": "Entity_name", "score": 0.4},
        {"id": "works_at", "collection": "EdgeType_relationship_name", "score": 0.2},
    ]
    assert payload["top_k"] == [
        {"source": "alice", "target": "acme", "rel": "works_at", "score": pytest.approx(0.6)}
    ]
    assert (payload["pool_size"], payload["pool_truncated"], payload["cut_size"]) == (3, False, 1)

    # No event of any kind is attributed to nothing.
    assert all(record["run_id"] is not None for record in sink.records)
    assert all(record["dataset_id"] == str(fakes.dataset.id) for record in sink.records)
    assert not capture.hook._buffer


@pytest.mark.asyncio
async def test_the_same_run_with_capture_off_is_a_structural_no_op(
    monkeypatch, tmp_path, pipeline_fakes
):
    """Capture off: the identical pipeline + search complete, the LLM fake is called the
    same number of times, and capture buffers nothing, starts no flusher, hashes nothing."""
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    fakes = pipeline_fakes
    user = SimpleNamespace(id=uuid4(), tenant_id=None, email="probe@example.test")

    def _boom(*args, **kwargs):
        raise AssertionError("prompt_fingerprint must not run while capture is off")

    monkeypatch.setattr(capture, "prompt_fingerprint", _boom)
    emit_spy = AsyncMock()  # never awaited; a plain call would surface as a MagicMock call
    monkeypatch.setattr(capture, "emit", emit_spy)

    async with record_operation_module.record_operation("remember") as remember_context:
        remember_context.set_dataset(fakes.dataset.id)
        events = await _run_pipeline(tmp_path, fakes, user)
    assert type(events[-1]).__name__ == "PipelineRunCompleted", events[-1]

    _search_context, search_results = await _run_search(fakes.dataset.id)
    assert len(search_results) == 1

    assert capture.is_active() is False
    assert capture.current_scope() is None
    emit_spy.assert_not_called()
    assert not capture.hook._buffer
    assert not capture.hook._flushers
    # 3 chunks x (extraction + summarization) LLM calls, same as with capture on.
    assert fakes.llm.await_count == 6
    # The stored summaries carry no provenance when capture is off.
    written_nodes = fakes.graph_engine.add_nodes.await_args.args[0]
    summaries = [node for node in written_nodes if node.type == "TextSummary"]
    assert len(summaries) == 3
    assert all(
        (node.model, node.prompt_fingerprint, node.source_text_hash) == (None, None, None)
        for node in summaries
    )


def _read_capture_dir(root):
    """Every record persisted under ``root`` by the StorageSink, keyed by relative path."""
    persisted = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            relative = os.path.relpath(path, root)
            if filename == "manifest.json":
                with open(path, encoding="utf-8") as handle:
                    persisted[relative] = [json.load(handle)]
            elif filename.endswith(".jsonl.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    persisted[relative] = [json.loads(line) for line in handle if line.strip()]
            else:
                raise AssertionError(f"unexpected file in the capture directory: {relative}")
    return persisted


@pytest.mark.asyncio
async def test_env_enabled_capture_persists_the_run_through_the_storage_sink(
    monkeypatch, tmp_path, pipeline_fakes
):
    """Capture switched on the way a deployment does it — ``COGNEE_CAPTURE_ENABLED`` and
    ``COGNEE_CAPTURE_DIR`` in the environment, no sink registered by hand — so the first
    ``is_active()`` auto-registers the real ``StorageSink`` over the local file storage,
    and one pipeline run plus one search land on disk in the documented layout::

        {dataset}/{run}/manifest.json
        {dataset}/{run}/{kind}/batch-*.jsonl.gz
    """
    capture_root = tmp_path / "capture"
    monkeypatch.setenv("COGNEE_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("COGNEE_CAPTURE_DIR", str(capture_root))
    fakes = pipeline_fakes
    user = SimpleNamespace(id=uuid4(), tenant_id=None, email="probe@example.test")

    async with record_operation_module.record_operation("remember") as remember_context:
        remember_context.set_dataset(fakes.dataset.id)
        events = await _run_pipeline(tmp_path, fakes, user)
    assert type(events[-1]).__name__ == "PipelineRunCompleted", events[-1]
    pipeline_run_id = str(events[0].pipeline_run_id)

    search_context, search_results = await _run_search(fakes.dataset.id)
    assert len(search_results) == 1

    assert isinstance(capture.hook._sink, capture.StorageSink), "env did not register the sink"
    await capture.drain()
    assert not capture.hook._buffer

    persisted = _read_capture_dir(capture_root)
    dataset = str(fakes.dataset.id)
    remember_run = str(remember_context.operation_id)
    search_run = str(search_context.operation_id)

    # Layout: three runs under the one dataset, nothing under nodataset/ or norun/.
    assert {path.split(os.sep)[0] for path in persisted} == {dataset}
    assert {path.split(os.sep)[1] for path in persisted} == {
        remember_run,
        pipeline_run_id,
        search_run,
    }

    def kinds_under(run_id):
        kinds = set()
        for path in persisted:
            parts = path.split(os.sep)
            if parts[1] != run_id:
                continue
            kinds.add("manifest.json" if parts[2] == "manifest.json" else parts[2])
        return kinds

    assert kinds_under(remember_run) == {"manifest.json"}
    assert kinds_under(pipeline_run_id) == {
        "manifest.json",
        KIND_EXTRACTION_CHUNK_GRAPH,
        KIND_EXTRACTION_DROPPED_DUPLICATES,
        KIND_EXTRACTION_FUZZY_MATCH,
        KIND_SUMMARY_GENERATED,
        KIND_STORAGE_DELTA,
    }
    assert kinds_under(search_run) == {"manifest.json", KIND_RETRIEVAL_CANDIDATES}

    # Every persisted record carries the full envelope and points at its own run.
    records = [record for group in persisted.values() for record in group]
    for record in records:
        assert set(record) == {"kind", "run_id", "dataset_id", "stage", "ts", "payload"}
        assert record["dataset_id"] == dataset
        _assert_flat(record["payload"])

    pipeline_manifest = persisted[os.path.join(dataset, pipeline_run_id, "manifest.json")][0]
    assert pipeline_manifest["kind"] == KIND_RUN_MANIFEST
    manifest = pipeline_manifest["payload"]
    assert manifest["parent_run_id"] == remember_run
    assert manifest["chunking.chunker"] == "TextChunker"
    assert manifest["extraction.model"] and manifest["extraction.prompt_fingerprint"]
    assert manifest["summarization.model"] and manifest["summarization.prompt_fingerprint"]
    assert manifest["ontology.mode"] == "annotate"
    assert manifest["counters"]["storage.nodes_written"] > 0
    assert manifest["dropped_events"] == 0

    chunk_graph_records = [
        record for record in records if record["kind"] == KIND_EXTRACTION_CHUNK_GRAPH
    ]
    assert len(chunk_graph_records) >= 2
    assert {record["run_id"] for record in chunk_graph_records} == {pipeline_run_id}
    [candidates] = [record for record in records if record["kind"] == KIND_RETRIEVAL_CANDIDATES]
    assert candidates["run_id"] == search_run
    assert candidates["payload"]["cut_size"] == 1
    search_manifest = persisted[os.path.join(dataset, search_run, "manifest.json")][0]
    assert search_manifest["payload"]["operation"] == "search"
    assert search_manifest["payload"]["outcome"] == "succeeded"

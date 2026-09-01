"""Extraction emit points for eval capture (SDK-529): the raw per-chunk graph is
snapshotted before dedup mutates it, dropped duplicate ids are reported once per
chunk, and the extraction model / prompt fingerprint and the ontology configuration
land on the run manifest — every one of them a structural no-op with capture off.

``find_existing_edge_identities`` is stubbed and ``calculate_chunk_graphs`` replaces
the LLM; the ``extract_content_graph`` tests stub ``LLMGateway``. No network, no
database.
"""

import importlib
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.context_global_variables import llm_config as llm_config_ctx
from cognee.infrastructure.llm.config import get_llm_config, get_llm_context_config
from cognee.modules.observability import capture
from cognee.modules.observability.capture import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_RUN_MANIFEST,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import extract_graph_from_data

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")
ecg_module = importlib.import_module(
    "cognee.infrastructure.llm.extraction.knowledge_graph.extract_content_graph"
)

pytestmark = pytest.mark.usefixtures("capture_reset")


class _UndumpableGraph(KnowledgeGraph):
    """A graph whose serialization must never be reached while capture is off."""

    def model_dump(self, *args, **kwargs):
        raise AssertionError("model_dump must not run while capture is off")


class _StubResolver(BaseOntologyResolver):
    """Matches nothing; carries a fuzzy strategy with a distinctive cutoff."""

    def __init__(self):
        super().__init__(FuzzyMatchingStrategy(cutoff=0.9))

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        return [], [], None


def _make_chunk(text="chunk text"):
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.text = text
    chunk.contains = None
    chunk.belongs_to_set = []
    return chunk


def _graph_with_duplicate(graph_class=KnowledgeGraph):
    return graph_class(
        nodes=[
            Node(id="dup", name="Alice", type="Person", description="first"),
            Node(id="dup", name="Bob", type="Person", description="second"),
            Node(id="n2", name="Carol", type="Person", description="third"),
        ],
        edges=[KGEdge(source_node_id="dup", target_node_id="n2", relationship_name="knows")],
    )


def _clean_graph(graph_class=KnowledgeGraph):
    return graph_class(
        nodes=[Node(id="n1", name="Dave", type="Person", description="desc")],
        edges=[],
    )


def _records(sink, kind):
    return [record for record in sink.records if record["kind"] == kind]


@pytest.fixture
def no_edge_lookup(monkeypatch):
    monkeypatch.setattr(egd_module, "find_existing_edge_identities", AsyncMock(return_value=set()))


async def _run_extraction(chunks, graphs, resolver=None, ontology_mode=None):
    async def fake_calc(data_chunks, graph_model, custom_prompt, **kwargs):
        return graphs

    ontology_config = {"ontology_resolver": resolver}
    if ontology_mode is not None:
        ontology_config["ontology_mode"] = ontology_mode

    return await extract_graph_from_data(
        chunks,
        type(graphs[0]),
        config={"ontology_config": ontology_config},
        calculate_chunk_graphs=fake_calc,
    )


# ---------------------------------------------------------------------------
# Off path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_off_never_dumps_buffers_or_starts_a_flusher(monkeypatch, no_edge_lookup):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    chunks = [_make_chunk(), _make_chunk()]
    graphs = [_graph_with_duplicate(_UndumpableGraph), _clean_graph(_UndumpableGraph)]

    await _run_extraction(chunks, graphs, resolver=_StubResolver())

    assert capture.is_active() is False
    assert not capture.hook._buffer
    assert not capture.hook._flushers
    # The dedup itself is unchanged: first node per id wins.
    assert [(node.id, node.name) for node in graphs[0].nodes] == [("dup", "Alice"), ("n2", "Carol")]


@pytest.mark.asyncio
async def test_extract_content_graph_off_path_never_hashes(monkeypatch):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    monkeypatch.setattr(
        ecg_module.LLMGateway, "acreate_structured_output", AsyncMock(return_value=_clean_graph())
    )

    def _boom(text):
        raise AssertionError("prompt_fingerprint must not run while capture is off")

    monkeypatch.setattr(capture, "prompt_fingerprint", _boom)

    result = await ecg_module.extract_content_graph("text", KnowledgeGraph, custom_prompt="p")

    assert isinstance(result, KnowledgeGraph)
    assert capture.current_scope() is None
    assert not capture.hook._buffer


# ---------------------------------------------------------------------------
# E1: raw per-chunk graph, snapshotted before dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_graph_payload_is_the_pre_dedup_snapshot(fake_capture_sink, no_edge_lookup):
    chunks = [_make_chunk("the first chunk"), _make_chunk("second")]
    graphs = [_graph_with_duplicate(), _clean_graph()]

    await _run_extraction(chunks, graphs)
    await capture.drain()

    events = _records(fake_capture_sink, KIND_EXTRACTION_CHUNK_GRAPH)
    assert [event["payload"]["chunk_index"] for event in events] == [0, 1]
    assert {event["stage"] for event in events} == {"extract_graph_from_data"}

    first = events[0]["payload"]
    assert first["chunk_id"] == str(chunks[0].id)
    assert first["chunk_size_chars"] == len("the first chunk")
    # Three nodes were extracted; the dedup that ran afterwards collapsed the live
    # graph to two — the snapshot still shows all three, in extraction order.
    assert [node["id"] for node in first["graph"]["nodes"]] == ["dup", "dup", "n2"]
    assert [node["name"] for node in first["graph"]["nodes"]] == ["Alice", "Bob", "Carol"]
    assert first["graph"]["edges"] == [
        {
            "source_node_id": "dup",
            "target_node_id": "n2",
            "relationship_name": "knows",
            "description": None,
        }
    ]
    assert [node.id for node in graphs[0].nodes] == ["dup", "n2"]
    # Plain JSON, no model references.
    json.dumps([event["payload"] for event in events])

    second = events[1]["payload"]
    assert second["chunk_id"] == str(chunks[1].id)
    assert second["chunk_size_chars"] == len("second")
    assert [node["id"] for node in second["graph"]["nodes"]] == ["n1"]


@pytest.mark.asyncio
async def test_chunk_graph_snapshot_survives_ontology_canonicalization(
    fake_capture_sink, no_edge_lookup
):
    class _Canonicalizer(_StubResolver):
        def get_subgraph(self, node_name, node_type="individuals", directed=True):
            from cognee.modules.ontology.models import AttachedOntologyNode

            if node_type == "individuals" and node_name == "dave":
                root = AttachedOntologyNode("https://example.test/onto#david", "individuals")
                return [root], [], root
            return [], [], None

    graphs = [_clean_graph()]

    await _run_extraction([_make_chunk()], graphs, resolver=_Canonicalizer())
    await capture.drain()

    [event] = _records(fake_capture_sink, KIND_EXTRACTION_CHUNK_GRAPH)
    # The live graph was renamed in place; the snapshot kept the extracted name.
    assert graphs[0].nodes[0].name == "david"
    assert event["payload"]["graph"]["nodes"][0]["name"] == "Dave"


# ---------------------------------------------------------------------------
# E6: dropped duplicates, one event per chunk graph that dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_dropped_duplicates_event_per_chunk_that_dropped(
    fake_capture_sink, no_edge_lookup
):
    chunks = [_make_chunk(), _make_chunk(), _make_chunk()]
    graphs = [
        _graph_with_duplicate(),
        _clean_graph(),
        KnowledgeGraph(
            nodes=[
                Node(id="x", name="Eve", type="Person", description="1"),
                Node(id="x", name="Frank", type="Person", description="2"),
                Node(id="x", name="Grace", type="Person", description="3"),
            ],
            edges=[],
        ),
    ]
    run_id = uuid4()

    with capture.run_scope(run_id, uuid4(), kind="pipeline"):
        await _run_extraction(chunks, graphs)
    await capture.drain()

    events = _records(fake_capture_sink, KIND_EXTRACTION_DROPPED_DUPLICATES)
    assert [
        (e["payload"]["chunk_index"], e["payload"]["dropped_node_ids"], e["payload"]["count"])
        for e in events
    ] == [(0, ["dup"], 1), (2, ["x", "x"], 2)]
    assert {event["run_id"] for event in events} == {str(run_id)}
    assert {event["stage"] for event in events} == {"extract_graph_from_data"}

    [manifest] = _records(fake_capture_sink, KIND_RUN_MANIFEST)
    assert manifest["payload"]["counters"]["extraction.dropped_duplicate_nodes"] == 3


@pytest.mark.asyncio
async def test_no_dropped_duplicates_event_when_nothing_was_dropped(
    fake_capture_sink, no_edge_lookup
):
    await _run_extraction([_make_chunk()], [_clean_graph()])
    await capture.drain()

    assert _records(fake_capture_sink, KIND_EXTRACTION_DROPPED_DUPLICATES) == []
    assert len(_records(fake_capture_sink, KIND_EXTRACTION_CHUNK_GRAPH)) == 1


# ---------------------------------------------------------------------------
# E5: ontology configuration on the manifest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_carries_the_ontology_configuration(fake_capture_sink, no_edge_lookup):
    with capture.run_scope(uuid4(), uuid4(), kind="pipeline"):
        await _run_extraction(
            [_make_chunk()], [_clean_graph()], resolver=_StubResolver(), ontology_mode="annotate"
        )
    await capture.drain()

    [manifest] = _records(fake_capture_sink, KIND_RUN_MANIFEST)
    payload = manifest["payload"]
    assert payload["ontology.mode"] == "annotate"
    assert payload["ontology.resolver"] == "_StubResolver"
    assert payload["ontology.matching_strategy"] == "FuzzyMatchingStrategy"
    assert payload["ontology.threshold"] == 0.9


@pytest.mark.asyncio
async def test_manifest_reports_a_run_without_an_ontology(fake_capture_sink, no_edge_lookup):
    with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
        await _run_extraction([_make_chunk()], [_clean_graph()], ontology_mode="strict")

    assert scope.fields["ontology.mode"] == "strict"
    assert scope.fields["ontology.resolver"] is None
    assert scope.fields["ontology.matching_strategy"] is None
    assert scope.fields["ontology.threshold"] is None


# ---------------------------------------------------------------------------
# E2: extraction model and prompt fingerprint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_content_graph_notes_model_and_prompt_fingerprint(
    monkeypatch, fake_capture_sink
):
    gateway = AsyncMock(return_value=_clean_graph())
    monkeypatch.setattr(ecg_module.LLMGateway, "acreate_structured_output", gateway)

    with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
        await ecg_module.extract_content_graph("text", KnowledgeGraph, custom_prompt="be brief")

    gateway.assert_awaited_once()
    assert scope.fields["extraction.prompt_fingerprint"] == capture.prompt_fingerprint("be brief")
    assert scope.fields["extraction.model"] == get_llm_context_config().llm_model


@pytest.mark.asyncio
async def test_extract_content_graph_notes_the_stage_routed_model(monkeypatch, fake_capture_sink):
    monkeypatch.setattr(
        ecg_module.LLMGateway, "acreate_structured_output", AsyncMock(return_value=_clean_graph())
    )
    routed = get_llm_config().model_copy(update={"llm_model": "test/extraction-model"})
    token = llm_config_ctx.set(routed)
    try:
        with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
            await ecg_module.extract_content_graph("text", KnowledgeGraph, custom_prompt="p")
    finally:
        llm_config_ctx.reset(token)

    assert scope.fields["extraction.model"] == "test/extraction-model"

"""Provenance stamps when two documents in one run share entities (GH #6498, SDK-1376).

An entity id derives from its name, so chunks of different documents that name the
same entity share one DataPoint object inside a batch, and that object is cross-linked
to both documents' entities. These tests pin, for that shape, what the batch stamp in
integrate_chunk_graphs must guarantee: every reachable DataPoint is stamped, each is
entered once, the pipeline context's visited set is untouched, and the same holds when
the pipeline runner drives the task and splits the documents over batches.
"""

import importlib
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.modules.pipelines.provenance_config import ProvenanceConfig
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import (
    _stamp_provenance_deep,
    extract_graph_from_data,
    integrate_chunk_graphs,
)

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")
runner_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_base")

NO_ONTOLOGY = {"ontology_config": {}}

# Two documents, two chunks each. "anna" and "petersburg" appear in both documents,
# "vasili" in two chunks of different documents, the rest in one chunk only.
GRAPHS = {
    "a1": (["anna", "vasili", "petersburg"], [("anna", "lives_in", "petersburg")]),
    "a2": (["pierre", "anna"], [("pierre", "visits", "anna")]),
    "b1": (["anna", "vasili", "helene"], [("vasili", "father_of", "helene")]),
    "b2": (["kutuzov", "petersburg"], [("kutuzov", "defends", "petersburg")]),
}
PLACES = {"petersburg"}


def _graph(key: str) -> KnowledgeGraph:
    names, edges = GRAPHS[key]
    nodes = [
        Node(id=n, name=n, type="Place" if n in PLACES else "Person", description="") for n in names
    ]
    return KnowledgeGraph(
        nodes=nodes,
        edges=[
            KGEdge(source_node_id=s, target_node_id=t, relationship_name=r) for s, r, t in edges
        ],
    )


def _document(name: str) -> TextDocument:
    return TextDocument(
        name=name, raw_data_location=f"/{name}", external_metadata="", mime_type="text/plain"
    )


def _chunk(document: TextDocument, index: int, key: str) -> DocumentChunk:
    return DocumentChunk(
        text=f"{key} text",
        chunk_size=2,
        chunk_index=index,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
    )


def _two_documents():
    """Chunks of A and B interleaved, as a mixed batch would hand them over."""
    a, b = _document("a"), _document("b")
    chunks = [_chunk(a, 0, "a1"), _chunk(b, 0, "b1"), _chunk(a, 1, "a2"), _chunk(b, 1, "b2")]
    return chunks, [_graph(c.text.split()[0]) for c in chunks]


def _entities(chunk: DocumentChunk) -> dict[str, DataPoint]:
    return {entity.name: entity for _, entity in chunk.contains}


def _reachable(roots) -> dict[int, DataPoint]:
    """Every DataPoint reachable from ``roots`` through model fields, keyed by id()."""
    found: dict[int, DataPoint] = {}
    stack = list(roots)
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, DataPoint) and id(item) not in found:
            found[id(item)] = item
            stack.extend(getattr(item, f, None) for f in item.model_fields)
    return found


@dataclass
class _FakeUser:
    id: str = str(uuid4())
    tenant_id: str = None
    email: str = "runner@example.com"


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_entities_shared_by_two_documents_are_one_object_stamped_once(
    mock_find_existing, monkeypatch
):
    mock_find_existing.return_value = set()
    calls, entered, original = [], [], egd_module._stamp_provenance_deep

    def spy(data, pipeline_name, task_name, visited=None):
        # The walk recurses through the module name, so the spy sees every call. A
        # DataPoint not yet in the set it arrives with is about to be entered: with
        # one set per batch that happens once per DataPoint, with a fresh set per
        # top-level call (the quadratic loop) once per path that reaches it.
        if isinstance(data, DataPoint) and id(data) not in (visited or ()):
            entered.append(id(data))
        if visited is None or visited is not (calls[-1] if calls else None):
            calls.append(visited)
        return original(data, pipeline_name, task_name, visited)

    monkeypatch.setattr(egd_module, "_stamp_provenance_deep", spy)
    ctx = PipelineContext(pipeline_name="cognify_pipeline")
    chunks, graphs = _two_documents()

    await integrate_chunk_graphs(
        chunks,
        graphs,
        KnowledgeGraph,
        None,
        pipeline_name="cognify_pipeline",
        task_name="extract_graph_from_data",
        ctx=ctx,
    )

    a1, b1, a2, b2 = (_entities(c) for c in chunks)
    # One object per entity across both documents, linked from every chunk naming it.
    assert a1["anna"] is b1["anna"] is a2["anna"]
    assert a1["petersburg"] is b2["petersburg"]
    assert a1["vasili"] is b1["vasili"]
    assert a1["anna"].is_a is b1["helene"].is_a, "one EntityType object for Person"

    reachable = _reachable([entity for c in chunks for _, entity in c.contains])
    unstamped = [
        dp.name
        for dp in reachable.values()
        if dp.source_pipeline != "cognify_pipeline" or dp.source_task != "extract_graph_from_data"
    ]
    assert unstamped == [], "every DataPoint reachable from any chunk must carry the stamp"
    assert calls and all(isinstance(v, set) and v is calls[0] for v in calls), (
        "one visited set, shared by the whole batch"
    )
    assert len(entered) == len(reachable), "each DataPoint is entered exactly once"
    assert ctx._provenance_visited == set(), "the runner's set is not touched here"


@pytest.fixture
def provenance_mode(monkeypatch):
    """Pin the mode the runner sees; ProvenanceConfig reads the env only at import."""

    def _set(mode: str):
        config = ProvenanceConfig(provenance_mode=mode)
        monkeypatch.setattr(runner_module, "get_provenance_config", lambda: config)

    return _set


async def _hand_over(chunks):
    """Stand-in for the chunking task: streams chunks like extract_chunks_from_documents,
    so the next task's batch_size decides how they are grouped, and the runner stamps
    each one and records it as visited before the graph task sees it."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lightweight", "deep"])
@pytest.mark.parametrize("batch_size", [4, 1], ids=["one_batch", "one_chunk_per_batch"])
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_runner_stamps_entities_of_two_documents(
    mock_find_existing, provenance_mode, mode, batch_size
):
    """Through the pipeline runner the chunks are already in ctx._provenance_visited
    when the graph task runs, so the runner never descends into the entities: the
    batch stamp is what must reach them, in every mode and however the batch splits."""
    mock_find_existing.return_value = set()
    provenance_mode(mode)
    chunks, graphs = _two_documents()
    graph_by_text = {c.text: g for c, g in zip(chunks, graphs)}

    def extractor(data_chunks, graph_model, custom_prompt, **kwargs):
        return [graph_by_text[c.text] for c in data_chunks]

    ctx = PipelineContext(pipeline_name="cognify_pipeline")
    tasks = [
        Task(_hand_over),
        Task(
            extract_graph_from_data,
            graph_model=KnowledgeGraph,
            config=NO_ONTOLOGY,
            calculate_chunk_graphs=extractor,
            task_config={"batch_size": batch_size},
        ),
    ]

    results = []
    async for result in run_tasks_base(tasks, chunks, _FakeUser(), ctx):
        results.extend(result if isinstance(result, list) else [result])

    assert sorted(c.text for c in results) == sorted(c.text for c in chunks)
    for chunk in results:
        # The runner's own stamps on the chunks, from the first task.
        assert chunk.source_task == "_hand_over"
        assert chunk.source_user == "runner@example.com"
        assert chunk.topological_rank == 1
        assert id(chunk) in ctx._provenance_visited

    entities = _reachable([entity for c in results for _, entity in c.contains])
    assert entities, "the extractor produced entities"
    unstamped = [
        dp.name
        for dp in entities.values()
        if dp.source_pipeline != "cognify_pipeline" or dp.source_task != "extract_graph_from_data"
    ]
    assert unstamped == []

    anna_objects = {id(_entities(c)["anna"]) for c in results if "anna" in _entities(c)}
    # One batch dedupes the shared entity into one object; one chunk per batch builds
    # it once per batch. Either way every copy is stamped (asserted above).
    assert len(anna_objects) == (1 if batch_size == 4 else 3)

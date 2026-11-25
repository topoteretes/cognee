"""Provenance stamping in integrate_chunk_graphs (GH #6498, SDK-1376).

The deep walk used to run once per constructed DataPoint with a fresh ``visited`` set.
The DataPoints are cross-linked by then, so every call re-walked the whole batch
(quadratic). The walk is the only stamp extracted entities get: the runner's pass
skips the chunks they hang off, already visited at the chunking stage.
"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.pipelines.models import PipelineContext
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import (
    _stamp_provenance_deep,
    integrate_chunk_graphs,
)

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")


def _chunk():
    chunk = MagicMock()
    chunk.text = "chunk text"
    chunk.contains = None
    chunk.belongs_to_set = []
    return chunk


def _hub_graph(entities: int = 6) -> KnowledgeGraph:
    """Entities linked to two shared hubs and back, so every node reaches every other."""
    hubs = [Node(id=f"h{j}", name=f"hub {j}", type="Place", description="") for j in range(2)]
    nodes = [
        Node(id=f"e{i}", name=f"entity {i}", type="Person", description="") for i in range(entities)
    ]
    edges = []
    for i, node in enumerate(nodes):
        for hub in (hubs[i % 2], hubs[(i + 1) % 2]):
            edges.append(
                KGEdge(source_node_id=node.id, target_node_id=hub.id, relationship_name="at")
            )
            edges.append(
                KGEdge(source_node_id=hub.id, target_node_id=node.id, relationship_name="hosts")
            )
    return KnowledgeGraph(nodes=nodes + hubs, edges=edges)


def _spy_stamping(monkeypatch):
    """Record every top-level stamping call's ``visited`` argument."""
    calls = []

    def spy(data, pipeline_name, task_name, visited=None):
        calls.append(visited)
        return _stamp_provenance_deep(data, pipeline_name, task_name, visited)

    monkeypatch.setattr(egd_module, "_stamp_provenance_deep", spy)
    return calls


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_batch_shares_one_visited_set(mock_find_existing, monkeypatch):
    mock_find_existing.return_value = set()
    calls = _spy_stamping(monkeypatch)

    chunk = _chunk()
    await integrate_chunk_graphs(
        [chunk], [_hub_graph()], KnowledgeGraph, None, pipeline_name="p", task_name="t"
    )

    assert len(calls) > 1, "one call per constructed DataPoint"
    assert all(v is calls[0] for v in calls), "every call must reuse the same visited set"
    assert isinstance(calls[0], set)
    stamped = [entity for _, entity in chunk.contains]  # contains holds (edge, entity) pairs
    assert stamped and all(dp.source_pipeline == "p" and dp.source_task == "t" for dp in stamped)


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_visited_set_is_not_the_pipeline_contexts(mock_find_existing, monkeypatch):
    """The runner skips ids in ctx._provenance_visited, so adding to it here would hide
    these DataPoints from any later runner pass that does reach them."""
    mock_find_existing.return_value = set()
    calls = _spy_stamping(monkeypatch)
    ctx = PipelineContext(pipeline_name="p")

    chunk = _chunk()
    await integrate_chunk_graphs(
        [chunk], [_hub_graph()], KnowledgeGraph, None, pipeline_name="p", task_name="t", ctx=ctx
    )

    assert calls and calls[0] is not ctx._provenance_visited
    assert ctx._provenance_visited == set()


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_no_pipeline_or_task_name_skips_the_walk(mock_find_existing, monkeypatch):
    mock_find_existing.return_value = set()
    calls = _spy_stamping(monkeypatch)

    chunk = _chunk()
    await integrate_chunk_graphs([chunk], [_hub_graph()], KnowledgeGraph, None)

    assert calls == []
    assert all(entity.source_pipeline is None for _, entity in chunk.contains)


def test_stamping_a_cross_linked_batch_is_linear_with_a_shared_visited_set():
    """The reporter's reproduction: with a shared set each DataPoint is entered once."""

    class Linked(DataPoint):
        name: str
        links: list["Linked"] = []

    Linked.model_rebuild()

    hubs = [Linked(name=f"h{j}") for j in range(2)]
    entities = [Linked(name=f"e{i}") for i in range(200)]
    for i, entity in enumerate(entities):
        entity.links = [hubs[i % 2], hubs[(i + 1) % 2]]
    for j, hub in enumerate(hubs):
        hub.links = [e for i, e in enumerate(entities) if j in (i % 2, (i + 1) % 2)]
    batch = entities + hubs

    class CountingSet(set):
        entered = 0

        def add(self, item):
            CountingSet.entered += 1
            super().add(item)

    shared = CountingSet()
    for data_point in batch:
        _stamp_provenance_deep(data_point, "p", "t", shared)

    assert CountingSet.entered == len(batch), "each DataPoint is visited exactly once"
    assert all(dp.source_pipeline == "p" for dp in batch)

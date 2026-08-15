from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.memify.cross_connect_entities import (
    INFERRED_EDGE_FEEDBACK_WEIGHT,
    InferredRelation,
    cross_connect_entities,
)

MODULE = "cognee.tasks.memify.cross_connect_entities"


class FakeGraph:
    def __init__(self, neighbors, existing_edges=None):
        self.neighbors = neighbors
        self.existing_edges = existing_edges or set()
        self.added_edges = None

    async def get_neighbors(self, node_id):
        return self.neighbors.get(node_id, [])

    async def has_edge(self, source_id, target_id, relationship_name):
        return (source_id, target_id, relationship_name) in self.existing_edges

    async def add_edges(self, edges):
        self.added_edges = edges


class FakeVector:
    def __init__(self, results_by_name=None, has_collection=True):
        self.results_by_name = results_by_name or {}
        self._has_collection = has_collection

    async def has_collection(self, collection_name):
        return self._has_collection

    async def search(self, collection_name, query_text=None, limit=None):
        return self.results_by_name.get(query_text, [])


def _entity(entity_id, name, description):
    return (entity_id, {"name": name, "description": description, "type": "Entity"})


def _patches(fake_graph, fake_vector, relation):
    return (
        patch(f"{MODULE}.get_graph_engine", new=AsyncMock(return_value=fake_graph)),
        patch(f"{MODULE}.get_vector_engine", new=MagicMock(return_value=fake_vector)),
        patch(
            f"{MODULE}.LLMGateway.acreate_structured_output",
            new=AsyncMock(return_value=relation),
        ),
        patch(f"{MODULE}.index_graph_edges", new=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_overlap_candidate_gets_inferred_edge():
    entities = [
        _entity("e1", "Ada Lovelace", "mathematician"),
        _entity("e2", "Analytical Engine", "mechanical computer"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    fake_graph = FakeGraph(
        {
            "e1": [{"id": "chunk_a"}],
            "e2": [{"id": "chunk_a"}],
            "e3": [{"id": "chunk_b"}],
        }
    )
    fake_vector = FakeVector(has_collection=False)
    relation = InferredRelation(related=True, relationship_name="runs_on", confidence=0.9)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p as index_mock:
        result = await cross_connect_entities(entities)

    assert result["written"] == 1
    assert result["proposed"] == [
        {"source": "e1", "target": "e2", "relationship_name": "runs_on", "confidence": 0.9}
    ]

    source, target, relationship, props = fake_graph.added_edges[0]
    assert (source, target, relationship) == ("e1", "e2", "runs_on")
    assert props["inferred"] is True
    assert props["feedback_weight"] == INFERRED_EDGE_FEEDBACK_WEIGHT
    index_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dry_run_lists_candidates_without_writing():
    entities = [
        _entity("e1", "Ada Lovelace", "mathematician"),
        _entity("e2", "Analytical Engine", "mechanical computer"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    fake_graph = FakeGraph(
        {"e1": [{"id": "chunk_a"}], "e2": [{"id": "chunk_a"}], "e3": [{"id": "chunk_b"}]}
    )
    fake_vector = FakeVector(has_collection=False)
    relation = InferredRelation(related=True, relationship_name="runs_on", confidence=0.9)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities, dry_run=True)

    assert result["dry_run"] is True
    assert result["written"] == 0
    assert len(result["proposed"]) == 1
    assert fake_graph.added_edges is None


@pytest.mark.asyncio
async def test_already_linked_pairs_are_skipped():
    entities = [
        _entity("e1", "Ada Lovelace", "mathematician"),
        _entity("e2", "Analytical Engine", "mechanical computer"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    # e1 and e2 share chunk_a and chunk_c (strong overlap candidate) but are already connected.
    fake_graph = FakeGraph(
        {
            "e1": [{"id": "chunk_a"}, {"id": "chunk_c"}, {"id": "e2"}],
            "e2": [{"id": "chunk_a"}, {"id": "chunk_c"}, {"id": "e1"}],
            "e3": [{"id": "chunk_b"}],
        }
    )
    fake_vector = FakeVector(has_collection=False)
    relation = InferredRelation(related=True, relationship_name="runs_on", confidence=0.9)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities)

    assert result["proposed"] == []
    assert fake_graph.added_edges is None


@pytest.mark.asyncio
async def test_has_edge_guard_drops_edge_with_stale_neighbors():
    entities = [
        _entity("e1", "Ada Lovelace", "mathematician"),
        _entity("e2", "Analytical Engine", "mechanical computer"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    # The adjacency view misses the link, but the graph already has the exact edge.
    fake_graph = FakeGraph(
        {"e1": [{"id": "chunk_a"}], "e2": [{"id": "chunk_a"}], "e3": [{"id": "chunk_b"}]},
        existing_edges={("e1", "e2", "runs_on")},
    )
    fake_vector = FakeVector(has_collection=False)
    relation = InferredRelation(related=True, relationship_name="runs_on", confidence=0.9)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities)

    assert result["written"] == 0
    assert fake_graph.added_edges == []


@pytest.mark.asyncio
async def test_low_confidence_relations_are_dropped():
    entities = [
        _entity("e1", "Ada Lovelace", "mathematician"),
        _entity("e2", "Analytical Engine", "mechanical computer"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    fake_graph = FakeGraph(
        {"e1": [{"id": "chunk_a"}], "e2": [{"id": "chunk_a"}], "e3": [{"id": "chunk_b"}]}
    )
    fake_vector = FakeVector(has_collection=False)
    relation = InferredRelation(related=True, relationship_name="runs_on", confidence=0.4)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities, confidence_threshold=0.7)

    assert result["proposed"] == []
    assert fake_graph.added_edges is None


@pytest.mark.asyncio
async def test_vector_similarity_candidate_is_linked():
    entities = [
        _entity("e1", "NYC", "New York City"),
        _entity("e2", "New York City", "largest US city"),
        _entity("e3", "Weather", "atmospheric conditions"),
    ]
    fake_graph = FakeGraph({"e1": [{"id": "a"}], "e2": [{"id": "b"}], "e3": [{"id": "c"}]})
    fake_vector = FakeVector(
        results_by_name={
            "NYC": [SimpleNamespace(id="e2", score=0.05)],
            "New York City": [SimpleNamespace(id="e1", score=0.05)],
        }
    )
    relation = InferredRelation(related=True, relationship_name="same_as", confidence=0.95)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities, similarity_threshold=0.5)

    assert result["written"] == 1
    assert result["proposed"][0]["source"] == "e1"
    assert result["proposed"][0]["target"] == "e2"


@pytest.mark.asyncio
async def test_max_new_edges_per_node_caps_growth():
    entities = [
        _entity("e1", "Hub", "central node"),
        _entity("e2", "Leaf one", "first leaf"),
        _entity("e3", "Leaf two", "second leaf"),
    ]
    # e1 is vector-similar to both e2 and e3; capping at 1 keeps only the first.
    fake_graph = FakeGraph({"e1": [{"id": "a"}], "e2": [{"id": "b"}], "e3": [{"id": "c"}]})
    fake_vector = FakeVector(
        results_by_name={
            "Hub": [SimpleNamespace(id="e2", score=0.05), SimpleNamespace(id="e3", score=0.05)],
        }
    )
    relation = InferredRelation(related=True, relationship_name="linked_to", confidence=0.9)

    graph_p, vector_p, llm_p, index_p = _patches(fake_graph, fake_vector, relation)
    with graph_p, vector_p, llm_p, index_p:
        result = await cross_connect_entities(entities, max_new_edges_per_node=1)

    assert result["written"] == 1

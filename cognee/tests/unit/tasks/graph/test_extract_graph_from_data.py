import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.shared.data_models import KnowledgeGraph, Node, Edge as KGEdge
from cognee.tasks.graph.extract_graph_from_data import (
    extract_graph_from_data,
    integrate_chunk_graphs,
)
from cognee.tasks.graph.exceptions import InvalidOntologyAdapterError

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")


def _mock_resolver():
    resolver = MagicMock()
    resolver.get_subgraph.return_value = ([], [], None)
    return resolver


def _make_chunk(text="chunk text"):
    chunk = MagicMock()
    chunk.text = text
    chunk.contains = None
    chunk.belongs_to_set = []
    return chunk


def _two_node_graph():
    return KnowledgeGraph(
        nodes=[
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        edges=[KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_integration_does_not_write_to_db(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()

    # If add_data_points were called it would fail (not mocked), proving it is not called.
    result = await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, _mock_resolver())

    assert result == [chunk]


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_chunk_contains_entities_after_integration(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, _mock_resolver())

    assert chunk.contains is not None and len(chunk.contains) > 0
    _, entity = chunk.contains[0]
    assert entity.name in ("alice", "bob")


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_entity_relations_populated_after_integration(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, _mock_resolver())

    entities = [e for _, e in chunk.contains]
    alice = next((e for e in entities if e.name == "alice"), None)
    assert alice is not None
    assert len(alice.relations) == 1
    _, target = alice.relations[0]
    assert target.name == "bob"


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_cache_entity_embeddings_hook_called(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Alice", type="Person", description="desc")],
        edges=[],
    )
    hook = MagicMock(return_value=None)

    await integrate_chunk_graphs(
        [chunk],
        [graph],
        KnowledgeGraph,
        _mock_resolver(),
        cache_entity_embeddings=hook,
    )

    hook.assert_called_once()
    entity_nodes_arg = hook.call_args.args[0]
    assert len(entity_nodes_arg) > 0


class _KGSubclass(KnowledgeGraph):
    pass


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_integrate_chunk_graphs_accepts_knowledge_graph_subclass(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _KGSubclass(
        nodes=[
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        edges=[KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )

    await integrate_chunk_graphs([chunk], [graph], _KGSubclass, _mock_resolver())

    # Subclass should take the integration path, not the contains-passthrough path.
    assert chunk.contains is not None and len(chunk.contains) > 0
    _, entity = chunk.contains[0]
    assert entity.name in ("alice", "bob")


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_extract_graph_keeps_dangling_edges_but_skips_them_during_integration(
    mock_find_existing,
):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _KGSubclass(
        nodes=[Node(id="n1", name="Alice", type="Person", description="desc")],
        edges=[
            KGEdge(source_node_id="n1", target_node_id="n1", relationship_name="self"),
            KGEdge(source_node_id="n1", target_node_id="missing", relationship_name="dangling"),
        ],
    )

    async def fake_calc(chunks, graph_model, custom_prompt, **kwargs):
        return [graph]

    config = {"ontology_config": {"ontology_resolver": _mock_resolver()}}

    await extract_graph_from_data(
        [chunk],
        _KGSubclass,
        config=config,
        calculate_chunk_graphs=fake_calc,
    )

    alice = next(entity for _, entity in chunk.contains if entity.name == "alice")
    assert len(graph.edges) == 2
    assert [(edge.relationship_type, target.name) for edge, target in alice.relations] == [
        ("self", "alice")
    ]


@pytest.mark.asyncio
async def test_non_knowledge_graph_model_unchanged():
    from pydantic import BaseModel

    class CustomModel(BaseModel):
        pass

    chunk = _make_chunk()
    custom_graph = CustomModel()

    result = await integrate_chunk_graphs([chunk], [custom_graph], CustomModel, _mock_resolver())

    assert chunk.contains == custom_graph
    assert result == [chunk]


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_integrate_chunk_graphs_accepts_none_resolver(mock_find_existing):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, None)

    assert chunk.contains is not None and len(chunk.contains) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("use_ontology", [False, True])
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_integrate_chunk_graphs_keeps_first_node_for_duplicate_extracted_id(
    mock_find_existing,
    use_ontology,
):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[
            Node(id="duplicate", name="Alice", type="Person", description="first"),
            Node(id="duplicate", name="Bob", type="Person", description="second"),
        ],
        edges=[
            KGEdge(
                source_node_id="duplicate",
                target_node_id="duplicate",
                relationship_name="knows",
            )
        ],
    )
    resolver = _mock_resolver() if use_ontology else None

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, resolver)

    assert [(node.id, node.name) for node in graph.nodes] == [("duplicate", "Alice")]
    assert len(chunk.contains) == 1
    _, alice = chunk.contains[0]
    assert alice.name == "alice"
    assert [(edge.relationship_type, target.name) for edge, target in alice.relations] == [
        ("knows", "alice")
    ]


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
@patch.object(egd_module, "construct_data_points_and_edges_with_ontology")
@patch.object(egd_module, "construct_data_points_and_edges")
async def test_integrate_chunk_graphs_selects_the_non_ontology_constructor(
    mock_construct,
    mock_construct_with_ontology,
    mock_find_existing,
):
    mock_construct.return_value = ({}, {})
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, None)

    mock_construct.assert_called_once_with([chunk], [graph])
    mock_construct_with_ontology.assert_not_called()


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
@patch.object(egd_module, "construct_data_points_and_edges_with_ontology")
@patch.object(egd_module, "construct_data_points_and_edges")
async def test_integrate_chunk_graphs_selects_the_ontology_constructor(
    mock_construct,
    mock_construct_with_ontology,
    mock_find_existing,
):
    mock_construct_with_ontology.return_value = ({}, {})
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()
    resolver = _mock_resolver()

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, resolver)

    mock_construct.assert_not_called()
    mock_construct_with_ontology.assert_called_once_with([chunk], [graph], resolver)


@pytest.mark.asyncio
async def test_integrate_chunk_graphs_rejects_invalid_resolver():
    chunk = _make_chunk()
    graph = _two_node_graph()

    with pytest.raises(InvalidOntologyAdapterError):
        await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, object())


@pytest.mark.asyncio
@patch.object(egd_module, "get_configured_ontology_resolver")
@patch.object(egd_module, "integrate_chunk_graphs", new_callable=AsyncMock)
async def test_extract_graph_from_data_passes_none_resolver(mock_integrate, mock_get_resolver):
    mock_get_resolver.return_value = None
    mock_integrate.side_effect = lambda *a, **kw: a[0]
    chunk = _make_chunk()

    async def fake_calc(chunks, graph_model, custom_prompt, **kwargs):
        return [_two_node_graph()]

    await extract_graph_from_data(
        [chunk],
        KnowledgeGraph,
        config={"ontology_config": {"ontology_resolver": None}},
        calculate_chunk_graphs=fake_calc,
    )

    assert mock_integrate.await_args.args[3] is None


@pytest.mark.asyncio
@patch.object(egd_module, "find_existing_edge_identities", new_callable=AsyncMock)
async def test_stub_resolver_reaches_graph_construction_via_task(mock_find_existing):
    from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
    from cognee.modules.ontology.models import AttachedOntologyNode

    class _TaskStubResolver(BaseOntologyResolver):
        def build_lookup(self) -> None:
            return None

        def refresh_lookup(self) -> None:
            return None

        def find_closest_match(self, name: str, category: str):
            return None

        def get_subgraph(
            self, node_name: str, node_type: str = "individuals", directed: bool = True
        ):
            if node_type == "classes" and node_name == "person":
                return (
                    [AttachedOntologyNode("person", "classes")],
                    [],
                    AttachedOntologyNode("person", "classes"),
                )
            if node_type == "individuals" and node_name == "alice":
                return (
                    [AttachedOntologyNode("alice", "individuals")],
                    [],
                    AttachedOntologyNode("alice", "individuals"),
                )
            return [], [], None

    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = KnowledgeGraph(
        nodes=[Node(id="n1", name="Alice", type="Person", description="desc")],
        edges=[],
    )
    resolver = _TaskStubResolver()

    await extract_graph_from_data(
        [chunk],
        KnowledgeGraph,
        config={"ontology_config": {"ontology_resolver": resolver}},
        calculate_chunk_graphs=lambda *args, **kwargs: [graph],
    )

    _, entity = chunk.contains[0]
    assert entity.name == "alice"
    assert entity.ontology_valid is True

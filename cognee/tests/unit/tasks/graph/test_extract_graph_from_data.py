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
async def test_integration_forwards_pipeline_context_for_existing_edge_provenance(
    mock_find_existing,
):
    mock_find_existing.return_value = set()
    chunk = _make_chunk()
    graph = _two_node_graph()
    ctx = MagicMock()

    await integrate_chunk_graphs(
        [chunk],
        [graph],
        KnowledgeGraph,
        _mock_resolver(),
        ctx=ctx,
    )

    mock_find_existing.assert_awaited_once()
    assert mock_find_existing.await_args.kwargs["ctx"] is ctx


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
    mock_construct_with_ontology.assert_called_once_with(
        [chunk], [graph], resolver, ontology_mode=None
    )


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


# --- chunk_attachment (SDK-163) -------------------------------------------------

from typing import Any, List, Optional  # noqa: E402

from cognee.infrastructure.engine import DataPoint  # noqa: E402
from cognee.modules.graph.utils import (  # noqa: E402
    ensure_default_edge_properties,
    get_graph_from_model,
)


class _Activity(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class _Person(DataPoint):
    name: str
    likes: Optional[List[_Activity]] = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class _Directory(DataPoint):
    people: List[_Person]
    metadata: dict = {"index_fields": []}


class _TransparentDirectory(DataPoint):
    people: List[_Person]
    metadata: dict = {"index_fields": [], "transparent": True}


class _AttachmentChunk(DataPoint):
    """A DocumentChunk-shaped node, so the real walk can mint the contains edges."""

    text: str
    contains: Any = None
    metadata: dict = {"index_fields": ["text"]}


def _directory_fixture(wrapper_class):
    biking = _Activity(name="Biking")
    basketball = _Activity(name="Basketball")
    alice = _Person(name="Alice", likes=[biking])
    bob = _Person(name="Bob", likes=[basketball])
    return wrapper_class(people=[alice, bob]), (alice, bob, biking, basketball)


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment", [None, "direct"])
async def test_direct_attachment_leaves_the_root_assignment_untouched(attachment):
    graph, _ = _directory_fixture(_Directory)
    chunk = _make_chunk()

    await integrate_chunk_graphs(
        [chunk], [graph], _Directory, _mock_resolver(), chunk_attachment=attachment
    )

    assert chunk.contains is graph


@pytest.mark.asyncio
async def test_all_attachment_lists_every_stored_node_in_order():
    graph, (alice, bob, biking, basketball) = _directory_fixture(_Directory)
    chunk = _make_chunk()

    await integrate_chunk_graphs(
        [chunk], [graph], _Directory, _mock_resolver(), chunk_attachment="all"
    )

    # A set: the walk's traversal order is not part of the contract.
    assert {str(node.id) for node in chunk.contains} == {
        str(graph.id),
        str(alice.id),
        str(bob.id),
        str(biking.id),
        str(basketball.id),
    }


@pytest.mark.asyncio
async def test_all_attachment_excludes_a_transparent_wrapper():
    graph, (alice, bob, biking, basketball) = _directory_fixture(_TransparentDirectory)
    chunk = _make_chunk()

    await integrate_chunk_graphs(
        [chunk], [graph], _TransparentDirectory, _mock_resolver(), chunk_attachment="all"
    )

    assert {str(node.id) for node in chunk.contains} == {
        str(alice.id),
        str(bob.id),
        str(biking.id),
        str(basketball.id),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper_class,attachment,expected_targets",
    [
        (_Directory, None, {"root"}),
        (_Directory, "direct", {"root"}),
        (_Directory, "all", {"root", "alice", "bob", "biking", "basketball"}),
        (_TransparentDirectory, None, {"alice", "bob"}),
        (_TransparentDirectory, "all", {"alice", "bob", "biking", "basketball"}),
    ],
)
async def test_attachment_matrix_over_the_real_walk(wrapper_class, attachment, expected_targets):
    graph, (alice, bob, biking, basketball) = _directory_fixture(wrapper_class)
    by_name = {
        "root": graph,
        "alice": alice,
        "bob": bob,
        "biking": biking,
        "basketball": basketball,
    }
    chunk = _AttachmentChunk(text="Alice likes biking. Bob likes basketball.")

    await integrate_chunk_graphs(
        [chunk], [graph], wrapper_class, _mock_resolver(), chunk_attachment=attachment
    )
    nodes, edges = await get_graph_from_model(chunk)
    edges = ensure_default_edge_properties(edges, nodes=nodes)

    contains = [edge for edge in edges if edge[2] == "contains"]
    assert {str(edge[1]) for edge in contains} == {
        str(by_name[name].id) for name in expected_targets
    }
    # One edge per target, and the shared policy supplied the label.
    assert len(contains) == len(expected_targets)
    assert all(edge[3].get("edge_text") for edge in contains)


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_class", [_Directory, _TransparentDirectory])
async def test_attachment_never_changes_a_non_contains_edge(wrapper_class):
    # One extraction, two chunks: chunk_attachment must only decide which nodes the
    # chunk links to, never touch the graph the model itself describes.
    graph, _ = _directory_fixture(wrapper_class)

    results = []
    for chunk_attachment in (None, "all"):
        chunk = _AttachmentChunk(text="Alice likes biking. Bob likes basketball.")
        await integrate_chunk_graphs(
            [chunk], [graph], wrapper_class, _mock_resolver(), chunk_attachment=chunk_attachment
        )
        _, edges = await get_graph_from_model(chunk)
        results.append({(str(s), str(t), name) for s, t, name, _ in edges if name != "contains"})

    assert results[0] == results[1]
    assert results[0]  # the model's own edges are actually present


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_graph", [None, "not-a-datapoint"])
async def test_all_attachment_falls_back_for_a_non_datapoint_extraction(chunk_graph):
    from pydantic import BaseModel

    class CustomModel(BaseModel):
        pass

    chunk = _make_chunk()

    await integrate_chunk_graphs(
        [chunk], [chunk_graph], CustomModel, _mock_resolver(), chunk_attachment="all"
    )

    assert chunk.contains is chunk_graph


@pytest.mark.asyncio
@patch.object(egd_module, "get_configured_ontology_resolver")
@patch.object(egd_module, "integrate_chunk_graphs", new_callable=AsyncMock)
@patch.object(egd_module, "extract_content_graph", new_callable=AsyncMock)
async def test_chunk_attachment_reaches_integration_by_keyword_only(
    mock_extract, mock_integrate, mock_get_resolver
):
    mock_get_resolver.return_value = None
    mock_integrate.side_effect = lambda *a, **kw: a[0]
    mock_extract.return_value = _two_node_graph()
    chunk = _make_chunk()

    await extract_graph_from_data([chunk], KnowledgeGraph, chunk_attachment="all")

    assert mock_integrate.await_args.kwargs["chunk_attachment"] == "all"
    # It must never reach the LLM call, which swallows unknown kwargs silently.
    assert "chunk_attachment" not in mock_extract.await_args.kwargs

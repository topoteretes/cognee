"""Unit tests for the consolidate_entities memify tasks and pipeline.

Everything here is mocked: no graph backend, no vector backend, no network.
The tests assert the consolidation *logic* (clustering, canonical selection,
direction-preserving edge re-pointing, duplicate + embedding deletion, the
dry_run no-op) and the pipeline *wiring* (tasks and config passed to memify).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from cognee.memify_pipelines.consolidate_entities import consolidate_entities_pipeline
from cognee.modules.engine.models.Entity import Entity
from cognee.tasks.memify.consolidate_entities import (
    _node_degrees,
    _pick_canonical,
    detect_entity_duplicates,
    merge_entity_duplicates,
    plan_edge_repointing,
)

GRAPH = "cognee.tasks.memify.consolidate_entities.get_graph_engine"
VECTOR = "cognee.tasks.memify.consolidate_entities.get_vector_engine"

# Deterministic, valid-UUID entity ids (Entity.id_for hashes the name).
ID_NYC = str(Entity.id_for("NYC"))
ID_NYCITY = str(Entity.id_for("New York City"))
ID_BANANA = str(Entity.id_for("Banana"))


def _graph_mock():
    graph = AsyncMock()
    graph.add_edge = AsyncMock()
    graph.add_nodes = AsyncMock()
    graph.delete_nodes = AsyncMock()
    return graph


def _vector_mock():
    vector = MagicMock()
    vector.embed_data = AsyncMock()
    vector.delete_data_points = AsyncMock()
    return vector


def _make_async_ctx_mock():
    """A MagicMock that behaves as an async-context-manager factory."""
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=inner)


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_detect_clusters_near_duplicate_entities():
    nodes = [
        (ID_NYC, {"name": "NYC", "type": "Entity"}),
        (ID_NYCITY, {"name": "New York City", "type": "Entity"}),
        (ID_BANANA, {"name": "Banana", "type": "Entity"}),
    ]
    edges = []  # no EntityType edges -> all entities share an (absent) type

    graph = _graph_mock()
    graph.get_graph_data = AsyncMock(return_value=(nodes, edges))
    vector = _vector_mock()
    # Returned in the same order as the names passed to embed_data.
    # NYC vs New York City -> cosine 0.96; Banana -> orthogonal.
    vector.embed_data = AsyncMock(
        return_value=[[1.0, 0.0, 0.0], [0.96, 0.28, 0.0], [0.0, 0.0, 1.0]]
    )

    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        result = await detect_entity_duplicates(None, config={"similarity_threshold": 0.85})

    clusters = result["clusters"]
    assert len(clusters) == 1
    assert {member["name"] for member in clusters[0]} == {"NYC", "New York City"}
    # Edges are forwarded for the merge step to reuse.
    assert result["edges"] == edges


@pytest.mark.asyncio
async def test_detect_respects_protect_node_types():
    nodes = [
        (ID_NYC, {"name": "NYC", "type": "Entity"}),
        (ID_NYCITY, {"name": "New York City", "type": "Entity"}),
        ("type-city", {"name": "City", "type": "EntityType"}),
    ]
    # Both entities are of EntityType "City" via an is_a-style edge.
    edges = [
        (ID_NYC, "type-city", "is_a", {}),
        (ID_NYCITY, "type-city", "is_a", {}),
    ]
    graph = _graph_mock()
    graph.get_graph_data = AsyncMock(return_value=(nodes, edges))
    vector = _vector_mock()
    vector.embed_data = AsyncMock(return_value=[[1.0, 0.0], [0.99, 0.14]])

    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        result = await detect_entity_duplicates(None, config={"protect_node_types": ["City"]})

    # "City" is protected, so neither entity is even considered.
    assert result["clusters"] == []
    vector.embed_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_detect_name_match_groups_punctuation_variants():
    # "USA" and "U.S.A." are not cosine-similar here, but their normalized names
    # are identical, so name_match (on by default) must still group them.
    id_usa = str(Entity.id_for("USA"))
    id_usa_dotted = str(Entity.id_for("U.S.A."))
    nodes = [
        (id_usa, {"name": "USA", "type": "Entity"}),
        (id_usa_dotted, {"name": "U.S.A.", "type": "Entity"}),
    ]

    graph = _graph_mock()
    graph.get_graph_data = AsyncMock(return_value=(nodes, []))
    vector = _vector_mock()
    vector.embed_data = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])  # cosine 0

    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        grouped = await detect_entity_duplicates(None, config={"similarity_threshold": 0.85})
        separate = await detect_entity_duplicates(
            None, config={"similarity_threshold": 0.85, "name_match": False}
        )

    assert len(grouped["clusters"]) == 1
    assert {m["name"] for m in grouped["clusters"][0]} == {"USA", "U.S.A."}
    # With name_match off and dissimilar vectors, they are left separate.
    assert separate["clusters"] == []


@pytest.mark.asyncio
async def test_detect_allow_cross_type_controls_cross_type_merge():
    # Two cosine-similar names that belong to DIFFERENT EntityTypes.
    nodes = [
        (ID_NYC, {"name": "NYC", "type": "Entity"}),
        (ID_NYCITY, {"name": "New York City", "type": "Entity"}),
        ("type-city", {"name": "City", "type": "EntityType"}),
        ("type-borough", {"name": "Borough", "type": "EntityType"}),
    ]
    edges = [
        (ID_NYC, "type-borough", "is_a", {}),
        (ID_NYCITY, "type-city", "is_a", {}),
    ]
    vectors = [[1.0, 0.0], [0.97, 0.24]]  # cosine ~0.97

    # Default: differing EntityTypes are NOT merged.
    graph = _graph_mock()
    graph.get_graph_data = AsyncMock(return_value=(nodes, edges))
    vector = _vector_mock()
    vector.embed_data = AsyncMock(return_value=vectors)
    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        default_result = await detect_entity_duplicates(None, config={"similarity_threshold": 0.85})
    assert default_result["clusters"] == []

    # allow_cross_type=True merges across types.
    graph = _graph_mock()
    graph.get_graph_data = AsyncMock(return_value=(nodes, edges))
    vector = _vector_mock()
    vector.embed_data = AsyncMock(return_value=vectors)
    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        cross = await detect_entity_duplicates(
            None, config={"similarity_threshold": 0.85, "allow_cross_type": True}
        )
    assert len(cross["clusters"]) == 1
    assert {m["name"] for m in cross["clusters"][0]} == {"NYC", "New York City"}


# --------------------------------------------------------------------------- #
# canonical selection
# --------------------------------------------------------------------------- #
def test_pick_canonical_prefers_higher_degree():
    cluster = [
        {"id": ID_NYC, "name": "NYC", "created_at": 50},
        {"id": ID_NYCITY, "name": "New York City", "created_at": 100},
    ]
    canonical, duplicates = _pick_canonical(cluster, {ID_NYC: 5, ID_NYCITY: 2})
    # NYC wins on degree even though New York City is older.
    assert canonical["id"] == ID_NYC
    assert [d["id"] for d in duplicates] == [ID_NYCITY]


def test_pick_canonical_breaks_degree_ties_by_age():
    cluster = [
        {"id": ID_NYC, "name": "NYC", "created_at": 200},
        {"id": ID_NYCITY, "name": "New York City", "created_at": 100},
    ]
    canonical, duplicates = _pick_canonical(cluster, {ID_NYC: 3, ID_NYCITY: 3})
    # Equal degree -> older created_at wins.
    assert canonical["id"] == ID_NYCITY
    assert [d["id"] for d in duplicates] == [ID_NYC]


def test_node_degrees_counts_both_endpoints():
    edges = [("a", "b", "r", {}), ("a", "c", "r", {}), ("d", "a", "r", {})]
    assert _node_degrees(edges) == {"a": 3, "b": 1, "c": 1, "d": 1}


# --------------------------------------------------------------------------- #
# edge re-pointing helper
# --------------------------------------------------------------------------- #
def test_plan_edge_repointing_preserves_direction_and_drops_self_loops():
    remap = {ID_NYC: ID_NYCITY}
    edges = [
        (ID_NYC, "country", "located_in", {}),  # duplicate as source
        ("visitor", ID_NYC, "visited", {}),  # duplicate as target
        (ID_NYCITY, "park", "has", {}),  # canonical edge, untouched
        (ID_NYC, ID_NYCITY, "same_as", {}),  # intra-cluster -> self-loop
    ]

    moved = plan_edge_repointing(edges, remap)

    assert (ID_NYCITY, "country", "located_in", {}) in moved  # source kept as source
    assert ("visitor", ID_NYCITY, "visited", {}) in moved  # target kept as target
    assert len(moved) == 2  # untouched canonical edge and the self-loop are excluded
    assert all(source != target for source, target, _, _ in moved)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _cluster_payload():
    canonical = {
        "id": ID_NYCITY,
        "name": "New York City",
        "created_at": 100,
        "type": None,
        "description": "Largest US city.",
        "props": {
            "id": ID_NYCITY,
            "name": "New York City",
            "type": "Entity",
            "description": "Largest US city.",
        },
    }
    duplicate = {
        "id": ID_NYC,
        "name": "NYC",
        "created_at": 200,
        "type": None,
        "description": "The Big Apple.",
        "props": {
            "id": ID_NYC,
            "name": "NYC",
            "type": "Entity",
            "description": "The Big Apple.",
        },
    }
    edges = [
        (ID_NYCITY, "country", "located_in", {}),  # canonical, untouched
        (ID_NYCITY, "park", "has", {}),  # canonical, untouched (degree tie-breaker)
        (ID_NYC, "subway", "operates", {}),  # duplicate as source
        ("tourist", ID_NYC, "visited", {}),  # duplicate as target
        (ID_NYC, ID_NYCITY, "same_as", {}),  # intra-cluster -> dropped
    ]
    return {"clusters": [[canonical, duplicate]], "edges": edges}


@pytest.mark.asyncio
async def test_merge_repoints_edges_and_deletes_duplicates():
    payload = _cluster_payload()
    graph = _graph_mock()
    vector = _vector_mock()

    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        result = await merge_entity_duplicates(payload, config={})

    # New York City and NYC tie on degree (3 each); New York City is older -> canonical.
    repointed = {
        (call.args[0], call.args[1], call.args[2]) for call in graph.add_edge.await_args_list
    }
    assert (ID_NYCITY, "subway", "operates") in repointed  # source remapped, direction kept
    assert ("tourist", ID_NYCITY, "visited") in repointed  # target remapped, direction kept
    assert graph.add_edge.await_count == 2  # untouched + self-loop edges are not re-added

    # Duplicate node and its embedding are removed.
    graph.delete_nodes.assert_awaited_once_with([ID_NYC])
    vector.delete_data_points.assert_awaited_once()
    collection, ids = vector.delete_data_points.await_args.args
    assert collection == "Entity_name"
    # ids are passed as strings (matches sibling deletion code, adapter-agnostic).
    assert ids == [ID_NYC]

    # Canonical persisted once, with a unioned description and unchanged id.
    graph.add_nodes.assert_awaited_once()
    persisted = graph.add_nodes.await_args.args[0]
    assert [entity.id for entity in persisted] == [UUID(ID_NYCITY)]
    assert "Largest US city." in persisted[0].description
    assert "The Big Apple." in persisted[0].description

    assert [entity.id for entity in result] == [UUID(ID_NYCITY)]


@pytest.mark.asyncio
async def test_merge_dry_run_performs_no_mutations():
    payload = _cluster_payload()
    graph = _graph_mock()
    vector = _vector_mock()

    graph_getter = AsyncMock(return_value=graph)
    with patch(GRAPH, new=graph_getter), patch(VECTOR, return_value=vector):
        result = await merge_entity_duplicates(payload, config={"dry_run": True})

    assert result == []
    # Not a single mutating call, and the engines are not even acquired.
    graph_getter.assert_not_called()
    graph.add_edge.assert_not_called()
    graph.add_nodes.assert_not_called()
    graph.delete_nodes.assert_not_called()
    vector.delete_data_points.assert_not_called()


@pytest.mark.asyncio
async def test_merge_without_clusters_is_noop():
    graph = _graph_mock()
    vector = _vector_mock()
    with patch(GRAPH, new=AsyncMock(return_value=graph)), patch(VECTOR, return_value=vector):
        result = await merge_entity_duplicates({"clusters": [], "edges": []}, config={})
    assert result == []
    graph.add_edge.assert_not_called()
    graph.delete_nodes.assert_not_called()


# --------------------------------------------------------------------------- #
# pipeline wiring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_wires_memify_tasks_and_config():
    user = MagicMock()
    user.id = "u1"
    dataset = SimpleNamespace(id="ds-1", owner_id="owner-1", name="main_dataset")

    module = "cognee.memify_pipelines.consolidate_entities"
    with (
        patch(f"{module}.get_default_user", new=AsyncMock(return_value=user)),
        patch(
            f"{module}.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(f"{module}.set_database_global_context_variables", new=_make_async_ctx_mock()) as ctx,
        patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock,
    ):
        result = await consolidate_entities_pipeline(
            similarity_threshold=0.9, dry_run=True, top_k=5, protect_node_types=["City"]
        )

    assert result == {"status": "ok"}
    # The graph is consolidated within the target dataset's database context.
    ctx.assert_called_once_with("ds-1", "owner-1")

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["data"] == [{}]
    assert kwargs["dataset"] == "ds-1"
    assert len(kwargs["extraction_tasks"]) == 1
    assert len(kwargs["enrichment_tasks"]) == 1

    detect_config = kwargs["extraction_tasks"][0].default_params["kwargs"]["config"]
    merge_config = kwargs["enrichment_tasks"][0].default_params["kwargs"]["config"]
    assert detect_config["similarity_threshold"] == 0.9
    assert detect_config["dry_run"] is True
    assert detect_config["top_k"] == 5
    assert detect_config["protect_node_types"] == ["City"]
    # detect and merge receive the same config object.
    assert merge_config == detect_config

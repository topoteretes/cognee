"""NodeSet-scoped depth retrieval against local graph and vector adapters."""

from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
import pytest_asyncio
from pydantic import BaseModel

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
    get_memory_fragment,
)


def _id(name):
    return str(uuid5(NAMESPACE_URL, f"cognee-scoped-depth-test:{name}"))


class _Payload(BaseModel):
    name: str
    belongs_to_set: list[str]


class _Embeddings:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest_asyncio.fixture
async def engines(tmp_path):
    graph = LadybugAdapter(
        db_path=str(tmp_path / "graph"),
        kuzu_num_threads=1,
        kuzu_buffer_pool_size=1 << 26,
        kuzu_max_db_size=1 << 28,
    )
    vector = LanceDBAdapter(str(tmp_path / "vectors"), None, _Embeddings())
    try:
        await graph.add_nodes(
            [
                SimpleNamespace(id=_id(name), name=name, type=node_type)
                for name, node_type in [
                    ("shared", "Entity"),
                    ("a-only", "Entity"),
                    ("b-only", "Entity"),
                    ("a-behind-b", "Entity"),
                    ("A", "NodeSet"),
                    ("B", "NodeSet"),
                ]
            ]
        )
        await graph.add_edges(
            [
                (_id(source), _id(target), relationship, {})
                for source, target, relationship in [
                    ("shared", "a-only", "related"),
                    ("shared", "b-only", "related"),
                    ("b-only", "a-behind-b", "related"),
                    ("shared", "A", "belongs_to"),
                    ("shared", "B", "belongs_to"),
                    ("a-only", "A", "belongs_to"),
                    ("a-behind-b", "A", "belongs_to"),
                    ("b-only", "B", "belongs_to"),
                ]
            ]
        )
        await vector.upsert_raw_vectors(
            "Seed_name",
            [
                {
                    "id": _id("shared"),
                    "vector": [1.0, 0.0, 0.0],
                    "payload": {"name": "shared", "belongs_to_set": ["A", "B"]},
                }
            ],
            payload_schema=_Payload,
        )
        yield SimpleNamespace(graph=graph, vector=vector)
    finally:
        await vector.close()
        await graph.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names,operator,depth,expected",
    [
        (["A"], "OR", 1, {"shared", "a-only", "A"}),
        (["A"], "OR", 2, {"shared", "a-only", "a-behind-b", "A"}),
        (["B"], "OR", 2, {"shared", "b-only", "B"}),
        (["A", "B"], "OR", 1, {"shared", "a-only", "b-only", "A", "B"}),
        (["A", "B"], "AND", 2, {"shared", "A", "B"}),
        (["missing"], "OR", 2, set()),
    ],
)
async def test_depth_projection_keeps_nodeset_scope(engines, names, operator, depth, expected):
    fragment = await get_memory_fragment(
        graph_engine=engines.graph,
        node_type=NodeSet,
        node_name=names,
        node_name_filter_operator=operator,
        relevant_ids_to_filter=[_id("shared")],
        neighborhood_depth=depth,
        feedback_influence=0.0,
    )

    assert {node.attributes["name"] for node in fragment.nodes.values()} == expected
    assert all(
        edge.node1.attributes["name"] in expected and edge.node2.attributes["name"] in expected
        for edge in fragment.edges
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [NodeSet, None])
@pytest.mark.parametrize(
    "names,operator,expected",
    [
        (["A"], "OR", {"shared", "a-only", "A"}),
        (["B"], "OR", {"shared", "b-only", "B"}),
        (["A", "B"], "OR", {"shared", "a-only", "b-only", "A", "B"}),
        (["A", "B"], "AND", {"shared", "A", "B"}),
    ],
)
async def test_scoped_depth_search_preserves_vector_seeds(
    engines, node_type, names, operator, expected
):
    results = await brute_force_triplet_search(
        query="shared",
        collections=["Seed_name"],
        unified_engine=engines,
        node_type=node_type,
        node_name=names,
        node_name_filter_operator=operator,
        neighborhood_depth=1,
        neighborhood_seed_top_k=1,
        top_k=100,
        feedback_influence=0.0,
    )

    assert results, "NodeSet filtering must not discard the neighborhood seeds"
    assert {
        node.attributes["name"] for edge in results for node in (edge.node1, edge.node2)
    } == expected


@pytest.mark.asyncio
async def test_scoped_depth_does_not_traverse_an_excluded_bridge(engines):
    fragment = CogneeGraph()
    await fragment.project_neighborhood_from_db(
        engines.graph,
        node_properties_to_project=["name"],
        edge_properties_to_project=[],
        seed_node_ids=[_id("shared")],
        node_name=["A"],
        edge_types=["related"],
        depth=2,
    )

    assert {node.attributes["name"] for node in fragment.nodes.values()} == {"shared", "a-only"}
    assert fragment.edges


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", ["shared", "a-only"])
async def test_scoped_depth_preserves_stored_edge_direction(engines, seed):
    fragment = await get_memory_fragment(
        graph_engine=engines.graph,
        node_type=NodeSet,
        node_name=["A"],
        relevant_ids_to_filter=[_id(seed)],
        neighborhood_depth=1,
        feedback_influence=0.0,
    )

    assert sorted(
        (
            edge.node1.attributes["name"],
            edge.attributes["relationship_type"],
            edge.node2.attributes["name"],
            edge.directed,
        )
        for edge in fragment.edges
    ) == [
        ("a-only", "belongs_to", "A", True),
        ("shared", "belongs_to", "A", True),
        ("shared", "related", "a-only", True),
    ]


@pytest.mark.asyncio
async def test_scoped_depth_preserves_real_reverse_edges_and_self_loops(engines):
    await engines.graph.add_edges(
        [
            (_id("a-only"), _id("shared"), "related", {}),
            (_id("shared"), _id("shared"), "related", {}),
        ]
    )
    fragment = await get_memory_fragment(
        graph_engine=engines.graph,
        node_type=NodeSet,
        node_name=["A"],
        relevant_ids_to_filter=[_id("shared")],
        neighborhood_depth=1,
        feedback_influence=0.0,
    )

    assert sorted(
        (
            edge.node1.attributes["name"],
            edge.attributes["relationship_type"],
            edge.node2.attributes["name"],
            edge.directed,
        )
        for edge in fragment.edges
    ) == [
        ("a-only", "belongs_to", "A", True),
        ("a-only", "related", "shared", True),
        ("shared", "belongs_to", "A", True),
        ("shared", "related", "a-only", True),
        ("shared", "related", "shared", True),
    ]


@pytest.mark.asyncio
async def test_depth_projection_respects_an_explicit_group_type(engines):
    class SyntheticGroup:
        pass

    await engines.graph.add_nodes([SimpleNamespace(id=_id("A"), name="A", type="SyntheticGroup")])
    fragment = await get_memory_fragment(
        graph_engine=engines.graph,
        node_type=SyntheticGroup,
        node_name=["A"],
        relevant_ids_to_filter=[_id("shared")],
        neighborhood_depth=1,
        feedback_influence=0.0,
    )

    assert {node.attributes["name"] for node in fragment.nodes.values()} == {
        "shared",
        "a-only",
        "A",
    }


@pytest.mark.asyncio
async def test_only_scoped_expansion_ids_are_rescored(engines, monkeypatch):
    rescored_ids = set()
    score_by_ids = engines.vector.score_by_ids

    async def record_scores(collection_name, data_point_ids, query_vector):
        rescored_ids.update(data_point_ids)
        return await score_by_ids(collection_name, data_point_ids, query_vector)

    monkeypatch.setattr(engines.vector, "score_by_ids", record_scores)
    results = await brute_force_triplet_search(
        query="shared",
        collections=["Seed_name"],
        unified_engine=engines,
        node_name=["A"],
        neighborhood_depth=1,
        top_k=100,
        feedback_influence=0.0,
    )

    assert results
    assert rescored_ids == {_id("a-only"), _id("A")}


@pytest.mark.asyncio
async def test_unfiltered_depth_search_keeps_both_scopes(engines):
    results = await brute_force_triplet_search(
        query="shared",
        collections=["Seed_name"],
        unified_engine=engines,
        neighborhood_depth=1,
        neighborhood_seed_top_k=1,
        top_k=100,
        feedback_influence=0.0,
    )

    assert {node.attributes["name"] for edge in results for node in (edge.node1, edge.node2)} == {
        "shared",
        "a-only",
        "b-only",
        "A",
        "B",
    }

"""Unit tests for HelixHybridAdapter against a mocked HelixClient.

No HelixDB server is required: ``HelixClient.query`` is replaced with an
AsyncMock so the tests assert the request shapes the adapter emits and how it
maps responses back into Cognee types (ScoredResult, node/edge tuples).
"""

import pytest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.hybrid.helix.HelixHybridAdapter import HelixHybridAdapter


class SimpleNode(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class FakeEmbeddingEngine:
    def __init__(self, dim=3):
        self.dim = dim

    async def embed_text(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def get_vector_size(self):
        return self.dim

    def get_batch_size(self):
        return 32


def make_adapter(query_side_effect=None):
    adapter = HelixHybridAdapter(
        base_url="http://localhost:6969", embedding_engine=FakeEmbeddingEngine()
    )
    # Skip the index-bootstrap round trip.
    adapter._initialized = True
    adapter.client.query = AsyncMock(side_effect=query_side_effect, return_value={})
    return adapter


def last_write(adapter):
    """Return (queries, returns) of the most recent write call."""
    for call in reversed(adapter.client.query.await_args_list):
        if call.kwargs.get("request_type") == "write":
            return call.kwargs["queries"], call.kwargs["returns"]
    raise AssertionError("no write call recorded")


# --------------------------------------------------------------------------- #
# Graph writes
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_query_raises_not_implemented():
    adapter = make_adapter()
    with pytest.raises(NotImplementedError):
        await adapter.query("MATCH (n) RETURN n")


@pytest.mark.asyncio
async def test_add_nodes_emits_upsert_triples():
    adapter = make_adapter()
    node = SimpleNode(id=uuid4(), name="Alice")

    await adapter.add_nodes([node])

    queries, _ = last_write(adapter)
    # existing-lookup + update (VarNotEmpty) + create (VarEmpty) = 3 entries.
    assert len(queries) == 3
    assert queries[0]["Query"]["name"] == "ex0"
    assert queries[1]["Query"]["condition"] == {"VarNotEmpty": "ex0"}
    assert queries[2]["Query"]["condition"] == {"VarEmpty": "ex0"}
    # create path uses AddN with the id + tenant baked in.
    add_n = queries[2]["Query"]["steps"][0]["AddN"]
    prop_names = [p[0] for p in add_n["properties"]]
    assert "id" in prop_names and "tenant_id" in prop_names


@pytest.mark.asyncio
async def test_add_edge_binds_target_then_adds_edge():
    adapter = make_adapter()
    src, tgt = str(uuid4()), str(uuid4())

    await adapter.add_edge(src, tgt, "KNOWS", {"weight": 1})

    queries, _ = last_write(adapter)
    assert queries[0]["Query"]["name"] == "e0_tgt"
    add_e = queries[1]["Query"]["steps"][-1]["AddE"]
    assert add_e["label"] == "KNOWS"
    assert add_e["to"] == {"Var": "e0_tgt"}
    edge_prop_names = [p[0] for p in add_e["properties"]]
    assert {"source_id", "target_id", "relationship_name", "tenant_id"} <= set(edge_prop_names)


@pytest.mark.asyncio
async def test_reads_are_tenant_scoped():
    adapter = HelixHybridAdapter(
        base_url="http://x", embedding_engine=FakeEmbeddingEngine(), tenant_id="ds-7"
    )
    adapter._initialized = True
    adapter.client.query = AsyncMock(return_value={"is_empty": 0})

    await adapter.is_empty()

    call = adapter.client.query.await_args
    nwhere = call.kwargs["queries"][0]["Query"]["steps"][0]["NWhere"]
    # With no extra predicate the tenant scope is a bare Eq on the read source.
    assert nwhere == {"Eq": ["tenant_id", {"String": "ds-7"}]}


@pytest.mark.asyncio
async def test_is_empty_true_and_false():
    adapter = make_adapter()
    adapter.client.query = AsyncMock(return_value={"is_empty": {"count": 0}})
    assert await adapter.is_empty() is True
    adapter.client.query = AsyncMock(return_value={"is_empty": {"count": 5}})
    assert await adapter.is_empty() is False


# --------------------------------------------------------------------------- #
# Graph reads
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_node_parses_value_map():
    nid = str(uuid4())
    adapter = make_adapter()
    adapter.client.query = AsyncMock(
        return_value={"node": {"properties": [{"id": nid, "name": "Bob", "$id": 7}]}}
    )

    node = await adapter.get_node(nid)
    # $id and other virtual fields are stripped.
    assert node == {"id": nid, "name": "Bob"}


@pytest.mark.asyncio
async def test_get_node_missing_returns_none():
    adapter = make_adapter()
    adapter.client.query = AsyncMock(return_value={"node": {"properties": []}})
    assert await adapter.get_node(str(uuid4())) is None


@pytest.mark.asyncio
async def test_get_connections_assembles_triples():
    a, b = str(uuid4()), str(uuid4())

    def side_effect(**kwargs):
        name = kwargs["returns"][0] if kwargs.get("returns") else None
        if name == "all_edges":
            return {
                "all_edges": {
                    "properties": [
                        {
                            "source_id": a,
                            "target_id": b,
                            "relationship_name": "KNOWS",
                            "tenant_id": "default",
                            "$id": 1,
                            "$from": 2,
                            "$to": 3,
                        }
                    ]
                }
            }
        if name == "nodes":
            return {"nodes": {"properties": [{"id": a, "name": "A"}, {"id": b, "name": "B"}]}}
        return {}

    adapter = make_adapter()
    adapter.client.query = AsyncMock(side_effect=side_effect)

    conns = await adapter.get_connections(a)
    assert len(conns) == 1
    source, edge, target = conns[0]
    assert source["id"] == a and target["id"] == b
    assert edge["relationship_name"] == "KNOWS"


# --------------------------------------------------------------------------- #
# Vector
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_data_points_embeds_and_writes_vector():
    adapter = make_adapter()
    node = SimpleNode(id=uuid4(), name="Alice")

    await adapter.create_data_points("SimpleNode_name", [node])

    queries, _ = last_write(adapter)
    # The create path carries the vector under the vec__-prefixed collection property.
    create_entry = queries[2]["Query"]["steps"][0]["AddN"]
    prop_names = [p[0] for p in create_entry["properties"]]
    assert "vec__SimpleNode_name" in prop_names
    vector_prop = next(p for p in create_entry["properties"] if p[0] == "vec__SimpleNode_name")
    assert "F32Array" in vector_prop[1]["Value"]


@pytest.mark.asyncio
async def test_search_maps_scored_results_with_distance():
    ids = [str(uuid4()), str(uuid4())]

    def side_effect(**kwargs):
        if kwargs["request_type"] == "read" and kwargs["returns"] == ["hits"]:
            return {
                "hits": {
                    "properties": [
                        {"id": ids[0], "distance": 0.1, "tenant_id": "default"},
                        {"id": ids[1], "distance": 0.4, "tenant_id": "default"},
                    ]
                }
            }
        return {}

    adapter = make_adapter(query_side_effect=side_effect)

    results = await adapter.search("Entity_name", query_text="hello", limit=2)
    assert all(isinstance(r, ScoredResult) for r in results)
    assert results[0].id == UUID(ids[0])
    assert results[0].score == 0.1
    assert results[0].payload is None  # include_payload defaults to False


@pytest.mark.asyncio
async def test_search_node_name_filter():
    keep, drop = str(uuid4()), str(uuid4())

    def side_effect(**kwargs):
        returns = kwargs.get("returns")
        if returns == ["hits"]:
            return {
                "hits": {
                    "properties": [
                        {"id": keep, "distance": 0.1, "tenant_id": "default"},
                        {"id": drop, "distance": 0.2, "tenant_id": "default"},
                    ]
                }
            }
        if returns == ["nodes"]:
            return {
                "nodes": {
                    "properties": [
                        {"id": keep, "belongs_to_set": ["setA"]},
                        {"id": drop, "belongs_to_set": ["setB"]},
                    ]
                }
            }
        return {}

    adapter = make_adapter(query_side_effect=side_effect)

    results = await adapter.search(
        "Entity_name", query_vector=[0.1, 0.2, 0.3], limit=5, node_name=["setA"]
    )
    assert [r.id for r in results] == [UUID(keep)]


@pytest.mark.asyncio
async def test_search_mutually_exclusive_params():
    from cognee.infrastructure.databases.exceptions import (
        MutuallyExclusiveQueryParametersError,
        MissingQueryParameterError,
    )

    adapter = make_adapter()
    with pytest.raises(MutuallyExclusiveQueryParametersError):
        await adapter.search("Entity_name", query_text="a", query_vector=[0.1])
    with pytest.raises(MissingQueryParameterError):
        await adapter.search("Entity_name")


@pytest.mark.asyncio
async def test_retrieve_returns_payload_scored_results():
    nid = str(uuid4())
    adapter = make_adapter()
    adapter.client.query = AsyncMock(
        return_value={"nodes": {"properties": [{"id": nid, "name": "X"}]}}
    )

    results = await adapter.retrieve("Entity_name", [nid])
    assert len(results) == 1
    assert results[0].id == UUID(nid)
    assert results[0].payload == {"id": nid, "name": "X"}
    assert results[0].score == 0


@pytest.mark.asyncio
async def test_has_edges_reads_count_dicts():
    # Per-edge Count responses come back as {"eN": {"count": N}}.
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    edges = [(a, b, "knows", {}), (a, c, "likes", {})]
    adapter = make_adapter()
    adapter.client.query = AsyncMock(return_value={"e0": {"count": 1}, "e1": {"count": 0}})

    existing = await adapter.has_edges(edges)
    assert existing == [edges[0]]


def test_node_data_strips_vector_props():
    from cognee.infrastructure.databases.graph.helix_driver.adapter import _node_data

    row = {"id": "x", "name": "A", "vec__Entity_name": [0.1, 0.2]}
    assert _node_data(row) == {"id": "x", "name": "A"}


@pytest.mark.asyncio
async def test_search_uses_prefixed_vector_property():
    recorded = {}

    def side_effect(**kwargs):
        if kwargs.get("returns") == ["hits"]:
            recorded["steps"] = kwargs["queries"][0]["Query"]["steps"]
            return {"hits": []}
        return {}

    adapter = make_adapter(query_side_effect=side_effect)
    await adapter.search("Entity_name", query_vector=[0.1, 0.2, 0.3], limit=3)

    vs = recorded["steps"][0]["VectorSearchNodes"]
    assert vs["property"] == "vec__Entity_name"


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_single_batch_has_vectors():
    adapter = make_adapter()
    node = SimpleNode(id=uuid4(), name="Alice")

    await adapter.add_nodes_with_vectors([node])

    # The final write batch upserts the node with its vector in one round trip.
    queries, _ = last_write(adapter)
    create_entry = next(
        q for q in queries if q["Query"]["steps"] and "AddN" in q["Query"]["steps"][0]
    )
    prop_names = [p[0] for p in create_entry["Query"]["steps"][0]["AddN"]["properties"]]
    assert "vec__SimpleNode_name" in prop_names

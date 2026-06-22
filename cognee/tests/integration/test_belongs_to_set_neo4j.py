"""Integration tests for Neo4j belongs_to_set upsert and detag.

Exercises the wire-level Cypher behavior of `Neo4jAdapter.add_nodes` and
`remove_belongs_to_set_tags` against a live Neo4j (with APOC) instance,
without touching LLMs or embeddings.

Skipped unless `TEST_NEO4J_URL` is set. Configure with the env vars
`TEST_NEO4J_URL`, `TEST_NEO4J_USERNAME`, and `TEST_NEO4J_PASSWORD` — point
them at whatever local Neo4j instance you're running.
"""

from __future__ import annotations

import os
from typing import List, Optional
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.engine import DataPoint

NEO4J_URL = os.getenv("TEST_NEO4J_URL")
NEO4J_USER = os.getenv("TEST_NEO4J_USERNAME") or os.getenv("TEST_NEO4J_USER")
NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD")

pytestmark = pytest.mark.skipif(
    not NEO4J_URL,
    reason="TEST_NEO4J_URL not set; skipping live Neo4j integration tests",
)


try:
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    HAS_NEO4J = True
except ModuleNotFoundError:
    HAS_NEO4J = False


class _TaggedPoint(DataPoint):
    """Minimal DataPoint used to exercise live Neo4j `belongs_to_set` semantics."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _fresh_adapter() -> "Neo4jAdapter":
    """Build and initialize a Neo4jAdapter against the live test instance."""
    adapter = Neo4jAdapter(
        graph_database_url=NEO4J_URL,
        graph_database_username=NEO4J_USER,
        graph_database_password=NEO4J_PASSWORD,
        graph_database_allow_anonymous=not NEO4J_USER,
    )
    await adapter.initialize()
    return adapter


async def _read_tag_property(adapter: "Neo4jAdapter", node_id: UUID) -> Optional[List[str]]:
    """Read a node's stored `belongs_to_set` property, returning `None` if the node is gone."""
    rows = await adapter.query(
        "MATCH (n {id: $id}) RETURN n.belongs_to_set AS tags",
        {"id": str(node_id)},
    )
    if not rows:
        return None
    return rows[0]["tags"]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_add_nodes_merges_belongs_to_set_across_calls():
    """Same id re-upserted with a new tag must union the tags on the Neo4j node property."""
    adapter = await _fresh_adapter()
    node_id = uuid4()

    try:
        await adapter.add_nodes(
            [_TaggedPoint(id=node_id, text="shared", belongs_to_set=["DatasetA"])]
        )
        await adapter.add_nodes(
            [_TaggedPoint(id=node_id, text="shared", belongs_to_set=["DatasetB"])]
        )

        tags = await _read_tag_property(adapter, node_id)
        assert tags is not None
        assert sorted(tags) == ["DatasetA", "DatasetB"]
    finally:
        await adapter.delete_nodes([str(node_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_add_nodes_merges_duplicate_ids_within_single_batch():
    """UNWIND on a batch with duplicate ids would otherwise recompute the
    union twice and the second SET can overwrite the first, losing tags
    only present on the earlier duplicate. The adapter dedupes in Python,
    unioning tags across duplicates before the query."""
    adapter = await _fresh_adapter()
    node_id = uuid4()

    try:
        await adapter.add_nodes(
            [
                _TaggedPoint(id=node_id, text="shared", belongs_to_set=["DatasetA"]),
                _TaggedPoint(id=node_id, text="shared", belongs_to_set=["DatasetB"]),
            ]
        )

        tags = await _read_tag_property(adapter, node_id)
        assert tags is not None
        assert sorted(tags) == ["DatasetA", "DatasetB"]
    finally:
        await adapter.delete_nodes([str(node_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_remove_belongs_to_set_tags_strips_property():
    """Detag must remove the target tag from every matching node's property array."""
    adapter = await _fresh_adapter()
    shared_id = uuid4()
    untouched_id = uuid4()

    try:
        await adapter.add_nodes(
            [
                _TaggedPoint(id=shared_id, text="shared", belongs_to_set=["Dev", "DevMirror"]),
                _TaggedPoint(id=untouched_id, text="untouched", belongs_to_set=["Production"]),
            ]
        )

        await adapter.remove_belongs_to_set_tags(["Dev"])

        assert sorted(await _read_tag_property(adapter, shared_id) or []) == ["DevMirror"]
        assert await _read_tag_property(adapter, untouched_id) == ["Production"]
    finally:
        await adapter.delete_nodes([str(shared_id), str(untouched_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_remove_belongs_to_set_tags_noop_for_empty_input():
    """Detag with no tags must leave stored properties unchanged."""
    adapter = await _fresh_adapter()
    node_id = uuid4()

    try:
        await adapter.add_nodes([_TaggedPoint(id=node_id, text="shared", belongs_to_set=["Dev"])])
        await adapter.remove_belongs_to_set_tags([])

        assert await _read_tag_property(adapter, node_id) == ["Dev"]
    finally:
        await adapter.delete_nodes([str(node_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_remove_belongs_to_set_tags_scoped_by_node_ids():
    """Scoped detag is the wire-level primitive behind the shared-data
    reconciliation: when a shared node loses one dataset's anchor while
    another node in the same dataset legitimately still owns the tag,
    we must only strip the tag from the targeted id, not globally."""
    adapter = await _fresh_adapter()
    targeted_id = uuid4()
    untouched_same_tag_id = uuid4()

    try:
        await adapter.add_nodes(
            [
                _TaggedPoint(id=targeted_id, text="shared", belongs_to_set=["alfa", "beta"]),
                _TaggedPoint(
                    id=untouched_same_tag_id,
                    text="mock_only",
                    belongs_to_set=["alfa"],
                ),
            ]
        )

        await adapter.remove_belongs_to_set_tags(["alfa"], node_ids=[str(targeted_id)])

        assert await _read_tag_property(adapter, targeted_id) == ["beta"]
        assert await _read_tag_property(adapter, untouched_same_tag_id) == ["alfa"]
    finally:
        await adapter.delete_nodes([str(targeted_id), str(untouched_same_tag_id)])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_remove_belongs_to_set_tags_prunes_edges_to_surviving_nodeset():
    """When the detag strips a tag from a shared node but the NodeSet
    itself survives (scoped detag path), the `belongs_to_set` edge
    from the node to that NodeSet becomes stale — the property says
    the node is no longer in the set, but the graph still shows an
    edge. The detag must delete those edges atomically so graph and
    property stay in sync."""
    adapter = await _fresh_adapter()
    node_id = uuid4()
    nodeset_id = uuid4()
    untouched_same_tag_id = uuid4()
    other_nodeset_id = uuid4()

    try:
        await adapter.query(
            "CREATE (:`__Node__`:NodeSet {id: $id, name: 'alfa'})",
            {"id": str(nodeset_id)},
        )
        await adapter.query(
            "CREATE (:`__Node__`:NodeSet {id: $id, name: 'beta'})",
            {"id": str(other_nodeset_id)},
        )

        await adapter.add_nodes(
            [
                _TaggedPoint(id=node_id, text="shared", belongs_to_set=["alfa", "beta"]),
                _TaggedPoint(
                    id=untouched_same_tag_id,
                    text="mock_only",
                    belongs_to_set=["alfa"],
                ),
            ]
        )
        await adapter.query(
            "MATCH (n {id: $nid}), (ns {id: $nsid}) CREATE (n)-[:belongs_to_set]->(ns)",
            {"nid": str(node_id), "nsid": str(nodeset_id)},
        )
        await adapter.query(
            "MATCH (n {id: $nid}), (ns {id: $nsid}) CREATE (n)-[:belongs_to_set]->(ns)",
            {"nid": str(node_id), "nsid": str(other_nodeset_id)},
        )
        await adapter.query(
            "MATCH (n {id: $nid}), (ns {id: $nsid}) CREATE (n)-[:belongs_to_set]->(ns)",
            {"nid": str(untouched_same_tag_id), "nsid": str(nodeset_id)},
        )

        await adapter.remove_belongs_to_set_tags(["alfa"], node_ids=[str(node_id)])

        assert await _read_tag_property(adapter, node_id) == ["beta"]
        assert await _read_tag_property(adapter, untouched_same_tag_id) == ["alfa"]

        targeted_edges = await adapter.query(
            "MATCH (n {id: $nid})-[r:belongs_to_set]->(ns:NodeSet) RETURN ns.name AS name",
            {"nid": str(node_id)},
        )
        assert [row["name"] for row in targeted_edges] == ["beta"], (
            "Stale belongs_to_set edge to alfa NodeSet must be pruned on the targeted node"
        )

        untouched_edges = await adapter.query(
            "MATCH (n {id: $nid})-[r:belongs_to_set]->(ns:NodeSet) RETURN ns.name AS name",
            {"nid": str(untouched_same_tag_id)},
        )
        assert [row["name"] for row in untouched_edges] == ["alfa"], (
            "Edge from an unscoped node to the same NodeSet must survive the scoped detag"
        )

        nodeset_rows = await adapter.query(
            "MATCH (ns:NodeSet {id: $id}) RETURN ns.name AS name", {"id": str(nodeset_id)}
        )
        assert [row["name"] for row in nodeset_rows] == ["alfa"]
    finally:
        await adapter.delete_nodes(
            [
                str(node_id),
                str(untouched_same_tag_id),
                str(nodeset_id),
                str(other_nodeset_id),
            ]
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_NEO4J, reason="neo4j extra not installed")
async def test_remove_belongs_to_set_tags_is_atomic(monkeypatch):
    """The two phases of the detag — property strip + edge prune — must
    share a transaction. If the edge prune fails, the property strip must
    roll back so the graph never observes a half-applied state where the
    tag is gone from the property but the stale `belongs_to_set` edge
    survives.

    Simulated by patching `Neo4jAdapter.query` to raise on the merged
    Cypher statement; the property must remain unchanged.
    """
    adapter = await _fresh_adapter()
    node_id = uuid4()
    nodeset_id = uuid4()

    try:
        await adapter.query(
            "CREATE (:`__Node__`:NodeSet {id: $id, name: 'alfa'})",
            {"id": str(nodeset_id)},
        )
        await adapter.add_nodes([_TaggedPoint(id=node_id, text="shared", belongs_to_set=["alfa"])])
        await adapter.query(
            "MATCH (n {id: $nid}), (ns {id: $nsid}) CREATE (n)-[:belongs_to_set]->(ns)",
            {"nid": str(node_id), "nsid": str(nodeset_id)},
        )

        # Sanity: pre-state has the tag and the edge.
        assert await _read_tag_property(adapter, node_id) == ["alfa"]

        # Force the merged detag query to fail. Whitelist the harness
        # queries (read-only MATCHes used by the test itself) so we can
        # still inspect post-state. The merged write touches `SET` +
        # `DELETE`, which the filter below catches.
        original_query = adapter.query

        async def failing_query(query, params=None):
            if "SET n.belongs_to_set" in query and "DELETE r" in query:
                raise RuntimeError("simulated edge-prune failure")
            return await original_query(query, params)

        monkeypatch.setattr(adapter, "query", failing_query)

        with pytest.raises(RuntimeError, match="simulated edge-prune failure"):
            await adapter.remove_belongs_to_set_tags(["alfa"], node_ids=[str(node_id)])

        monkeypatch.undo()

        # Property must still carry the tag — the failed transaction rolled back.
        assert await _read_tag_property(adapter, node_id) == ["alfa"], (
            "Property strip must roll back when the edge prune fails"
        )

        # Edge must also still exist for the same reason.
        edges = await adapter.query(
            "MATCH (n {id: $nid})-[:belongs_to_set]->(ns:NodeSet) RETURN ns.name AS name",
            {"nid": str(node_id)},
        )
        assert [row["name"] for row in edges] == ["alfa"], (
            "Edge must survive the failed transaction"
        )
    finally:
        await adapter.delete_nodes([str(node_id), str(nodeset_id)])

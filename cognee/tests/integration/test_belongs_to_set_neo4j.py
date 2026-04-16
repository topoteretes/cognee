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
    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _fresh_adapter() -> "Neo4jAdapter":
    adapter = Neo4jAdapter(
        graph_database_url=NEO4J_URL,
        graph_database_username=NEO4J_USER,
        graph_database_password=NEO4J_PASSWORD,
        graph_database_allow_anonymous=not NEO4J_USER,
    )
    await adapter.initialize()
    return adapter


async def _read_tag_property(adapter: "Neo4jAdapter", node_id: UUID) -> Optional[List[str]]:
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
    adapter = await _fresh_adapter()
    node_id = uuid4()

    try:
        await adapter.add_nodes([_TaggedPoint(id=node_id, text="shared", belongs_to_set=["Dev"])])
        await adapter.remove_belongs_to_set_tags([])

        assert await _read_tag_property(adapter, node_id) == ["Dev"]
    finally:
        await adapter.delete_nodes([str(node_id)])

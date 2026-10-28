"""Regression test: LadybugAdapter.add_node (singular) upsert on the default backend.

The singular ``add_node`` built its MERGE with the map-literal form
``ON CREATE SET n += {{...}}``, which does not parse on the Kuzu/Ladybug backend
(``Parser exception``) — so every call crashed, breaking callers such as the web
scraper on the default graph backend. It also had no ``ON MATCH SET`` clause, so
even where it did run it silently dropped updates to existing nodes (the batch
``add_nodes`` already assigns each property individually with both ON CREATE and
ON MATCH).

``add_node`` now delegates to ``add_nodes``, whose MERGE overwrites name, type and
properties on match, keeps ``created_at`` and refreshes ``updated_at``. This test
adds a node, then re-adds it with a changed name and type, and asserts exactly
that contract — so a later edit of the ``add_nodes`` MERGE that sets ``created_at``
in ON MATCH, or drops ``updated_at`` (or the whole branch) from it, fails here.
"""

import asyncio
import contextlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.engine import DataPoint


class _Node(DataPoint):
    name: str
    description: str = ""
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture
async def adapter(tmp_path):
    a = LadybugAdapter(db_path=str(tmp_path / "ladybug_add_node"))
    if hasattr(a, "initialize"):
        await a.initialize()
    try:
        yield a
    finally:
        with contextlib.suppress(Exception):
            await a.close()


async def _timestamps(adapter, node_id):
    """The declared TIMESTAMP columns; get_node does not return them."""
    rows = await adapter.query(
        "MATCH (n:Node {id: $id}) RETURN n.created_at, n.updated_at", {"id": str(node_id)}
    )
    assert len(rows) == 1
    return rows[0][0], rows[0][1]


@pytest.mark.asyncio
async def test_add_node_creates_then_updates(adapter):
    node_id = uuid4()

    # Create — must not raise a parser exception.
    await adapter.add_node(_Node(id=node_id, name="OLD", description="first"))
    created = await adapter.get_node(node_id)
    assert created is not None
    assert created["name"] == "OLD"
    assert created["description"] == "first"
    # The type column is the DataPoint class name, so it is stable across re-adds.
    assert created["type"] == "_Node"
    created_at, first_updated_at = await _timestamps(adapter, node_id)
    assert created_at is not None
    assert first_updated_at == created_at

    # The write stamps are taken per call at microsecond resolution; make the
    # second call land measurably later so "refreshed" is a strict comparison.
    await asyncio.sleep(0.01)

    # Re-add with a changed name and property — the overwrite must persist
    # (ON MATCH SET on name and the properties blob), created_at must survive
    # and updated_at must move.
    await adapter.add_node(_Node(id=node_id, name="NEW", description="second"))
    updated = await adapter.get_node(node_id)
    assert updated is not None
    assert updated["name"] == "NEW"
    assert updated["description"] == "second"
    assert updated["type"] == "_Node"
    created_at_after, second_updated_at = await _timestamps(adapter, node_id)
    assert created_at_after == created_at
    assert second_updated_at > first_updated_at

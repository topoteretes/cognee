"""Regression test: LadybugAdapter.add_node (singular) upsert on the default backend.

The singular ``add_node`` built its MERGE with the map-literal form
``ON CREATE SET n += {{...}}``, which does not parse on the Kuzu/Ladybug backend
(``Parser exception``) — so every call crashed, breaking callers such as the web
scraper on the default graph backend. It also had no ``ON MATCH SET`` clause, so
even where it did run it silently dropped updates to existing nodes (the batch
``add_nodes`` already assigns each property individually with both ON CREATE and
ON MATCH).

This test adds a node, then re-adds it with a changed name, and asserts the
update persists — which both exercises the per-property MERGE syntax (no parser
crash) and the ON MATCH update.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


class _Node(DataPoint):
    name: str
    type: str = "Entity"
    metadata: dict = {"index_fields": ["name"]}


@pytest_asyncio.fixture
async def adapter(tmp_path):
    a = LadybugAdapter(db_path=str(tmp_path / "ladybug_add_node"))
    if hasattr(a, "initialize"):
        await a.initialize()
    try:
        yield a
    finally:
        try:
            await a.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_add_node_creates_then_updates(adapter):
    node_id = uuid4()

    # Create — must not raise a parser exception.
    await adapter.add_node(_Node(id=node_id, name="OLD"))
    created = await adapter.get_node(node_id)
    assert created is not None
    assert created["name"] == "OLD"

    # Re-add with a changed name — the update must persist (ON MATCH SET).
    await adapter.add_node(_Node(id=node_id, name="NEW"))
    updated = await adapter.get_node(node_id)
    assert updated is not None
    assert updated["name"] == "NEW"

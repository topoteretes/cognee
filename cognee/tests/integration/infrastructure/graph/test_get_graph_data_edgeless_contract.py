"""Edgeless-graph contract for ``get_graph_data()`` across embedded backends.

Every backend must return the real (empty) edge list when a graph has nodes
and no edges. Ladybug used to fabricate one ``(id, id, "SELF")`` edge per
node, which leaked into retrieval, the edge vector index, code-graph dedup
and entity consolidation. See issue #5042.
"""

from __future__ import annotations

import pytest

from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

try:
    from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter

    HAS_TURSO = True
except ModuleNotFoundError:
    HAS_TURSO = False

pytestmark = pytest.mark.asyncio


class _Node(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


async def test_ladybug_edgeless_graph_returns_no_edges(tmp_path):
    if not HAS_LADYBUG:
        pytest.skip("ladybug not installed")

    adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    try:
        await adapter.add_nodes([_Node(name="alice"), _Node(name="bob")])

        nodes, edges = await adapter.get_graph_data()

        assert len(nodes) == 2
        assert edges == []
    finally:
        await adapter.delete_graph()


async def test_ladybug_edges_still_returned_when_present(tmp_path):
    """The fix removes fabrication, not real edges."""
    if not HAS_LADYBUG:
        pytest.skip("ladybug not installed")

    adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    try:
        alice, bob = _Node(name="alice"), _Node(name="bob")
        await adapter.add_nodes([alice, bob])
        await adapter.add_edge(
            str(alice.id),
            str(bob.id),
            "knows",
            {"relationship_name": "knows"},
        )

        nodes, edges = await adapter.get_graph_data()

        assert len(nodes) == 2
        assert [(edge[0], edge[1], edge[2]) for edge in edges] == [
            (str(alice.id), str(bob.id), "knows")
        ]
    finally:
        await adapter.delete_graph()


async def test_turso_edgeless_graph_returns_no_edges():
    """Parity witness: the same graph on turso already returns zero edges."""
    if not HAS_TURSO:
        pytest.skip("turso not installed")

    adapter = TursoAdapter(connection_string="sqlite+aiosqlite:///:memory:")
    await adapter.initialize()
    try:
        await adapter.add_nodes([_Node(name="alice"), _Node(name="bob")])

        nodes, edges = await adapter.get_graph_data()

        assert len(nodes) == 2
        assert edges == []
    finally:
        await adapter.delete_graph()
        await adapter.close()

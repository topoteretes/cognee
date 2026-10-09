"""When delete_from_graph_and_vector processes a dataset whose uniquely-owned
NodeSet nodes are in the hard-delete set, it must also detag the deleted
NodeSet's name from every surviving row/node. This test exercises the
orchestration wiring without needing a live Neo4j or Postgres.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.modules.graph.models.EdgeType import EdgeType

delete_from_graph_and_vector_module = importlib.import_module(
    "cognee.modules.graph.methods.delete_from_graph_and_vector"
)
delete_from_graph_and_vector = delete_from_graph_and_vector_module.delete_from_graph_and_vector


def _node(node_type: str, label: str | None = None):
    """Build a minimal Node-ledger stand-in with the attributes the orchestrator reads."""
    return SimpleNamespace(
        slug=uuid4(),
        type=node_type,
        label=label,
        indexed_fields=["name"],
    )


@pytest.mark.asyncio
async def test_detag_called_with_removed_nodeset_names():
    """A NodeSet entry in `unique_nodes` must trigger detag on both engines."""
    nodeset_node = _node("NodeSet", label="Dev")
    entity_node = _node("Entity", label="Alice")
    nodes = [nodeset_node, entity_node]

    graph_engine = AsyncMock()
    vector_engine = AsyncMock()

    with (
        patch.object(
            delete_from_graph_and_vector_module,
            "get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "get_vector_engine_async",
            AsyncMock(return_value=vector_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_nodes_as_deleted",
            AsyncMock(),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_edges_as_deleted",
            AsyncMock(),
        ),
    ):
        await delete_from_graph_and_vector(
            affected_nodes=nodes,
            affected_edges=[],
            is_legacy_node=[False, False],
            is_legacy_edge=[],
        )

    graph_engine.remove_belongs_to_set_tags.assert_awaited_once_with(["Dev"])
    vector_engine.remove_belongs_to_set_tags.assert_awaited_once_with(["Dev"])


@pytest.mark.asyncio
async def test_detag_skipped_when_no_nodeset_in_batch():
    """No NodeSet in `unique_nodes` means detag must not be called."""
    entity_node = _node("Entity", label="Alice")
    nodes = [entity_node]

    graph_engine = AsyncMock()
    vector_engine = AsyncMock()

    with (
        patch.object(
            delete_from_graph_and_vector_module,
            "get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "get_vector_engine_async",
            AsyncMock(return_value=vector_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_nodes_as_deleted",
            AsyncMock(),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_edges_as_deleted",
            AsyncMock(),
        ),
    ):
        await delete_from_graph_and_vector(
            affected_nodes=nodes,
            affected_edges=[],
            is_legacy_node=[False],
            is_legacy_edge=[],
        )

    graph_engine.remove_belongs_to_set_tags.assert_not_awaited()
    vector_engine.remove_belongs_to_set_tags.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_uses_edge_text_for_edge_type_delete_ids():
    """Natural-language EdgeType rows are keyed by edge_text, not relationship_name."""
    edge = SimpleNamespace(
        slug=generate_edge_id("works_at"),
        source_node_id=uuid4(),
        destination_node_id=uuid4(),
        relationship_name="works_at",
        attributes={"edge_text": "Alice works at Acme."},
    )

    graph_engine = AsyncMock()
    graph_engine.get_graph_data.return_value = ([], [])
    vector_engine = AsyncMock()

    with (
        patch.object(
            delete_from_graph_and_vector_module,
            "get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "get_vector_engine_async",
            AsyncMock(return_value=vector_engine),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_nodes_as_deleted",
            AsyncMock(),
        ),
        patch.object(
            delete_from_graph_and_vector_module,
            "mark_ledger_edges_as_deleted",
            AsyncMock(),
        ),
    ):
        await delete_from_graph_and_vector(
            affected_nodes=[],
            affected_edges=[edge],
            is_legacy_node=[],
            is_legacy_edge=[False],
        )

    vector_engine.delete_data_points.assert_any_await(
        "EdgeType_relationship_name",
        [str(EdgeType.id_for("Alice works at Acme."))],
    )
    graph_engine.delete_nodes.assert_awaited_once_with(
        [str(EdgeType.id_for("Alice works at Acme."))]
    )

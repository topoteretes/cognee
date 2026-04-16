"""When delete_from_graph_and_vector processes a dataset whose uniquely-owned
NodeSet nodes are in the hard-delete set, it must also detag the deleted
NodeSet's name from every surviving row/node. This test exercises the
orchestration wiring without needing a live Neo4j or Postgres.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.graph.methods.delete_from_graph_and_vector import (
    delete_from_graph_and_vector,
)


def _node(node_type: str, label: str | None = None):
    return SimpleNamespace(
        slug=uuid4(),
        type=node_type,
        label=label,
        indexed_fields=["name"],
    )


@pytest.mark.asyncio
async def test_detag_called_with_removed_nodeset_names():
    nodeset_node = _node("NodeSet", label="Dev")
    entity_node = _node("Entity", label="Alice")
    nodes = [nodeset_node, entity_node]

    graph_engine = AsyncMock()
    vector_engine = AsyncMock()

    with (
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.get_vector_engine",
            lambda: vector_engine,
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.mark_ledger_nodes_as_deleted",
            AsyncMock(),
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.mark_ledger_edges_as_deleted",
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
    entity_node = _node("Entity", label="Alice")
    nodes = [entity_node]

    graph_engine = AsyncMock()
    vector_engine = AsyncMock()

    with (
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.get_graph_engine",
            AsyncMock(return_value=graph_engine),
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.get_vector_engine",
            lambda: vector_engine,
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.mark_ledger_nodes_as_deleted",
            AsyncMock(),
        ),
        patch(
            "cognee.modules.graph.methods.delete_from_graph_and_vector.mark_ledger_edges_as_deleted",
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

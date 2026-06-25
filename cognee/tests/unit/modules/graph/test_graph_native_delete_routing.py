"""Routing seam tests for graph-native delete (COG-5522).

Prove that the public delete entry points send graph-native graphs through the
unified boundary (and old/unmarked graphs stay on the relational-ledger path).
These mock the unified engine + marker so they stay fast and backend-free; the
real end-to-end behavior is covered by the integration suite.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# The package __init__ re-exports these functions under the same name as their
# submodule, so `import a.b.c as x` would bind the function, not the module.
# Pull the real module objects from sys.modules instead.
import cognee.modules.graph.methods.delete_data_nodes_and_edges  # noqa: F401
import cognee.modules.graph.methods.delete_dataset_nodes_and_edges  # noqa: F401
from cognee.infrastructure.databases.provenance import make_source_ref_key

ddne_module = sys.modules["cognee.modules.graph.methods.delete_data_nodes_and_edges"]
ddsne_module = sys.modules["cognee.modules.graph.methods.delete_dataset_nodes_and_edges"]

pytestmark = pytest.mark.asyncio


def _unified(graph_native_supported=True):
    return SimpleNamespace(
        supports_graph_native_delete=lambda: graph_native_supported,
        graph=object(),
        delete_by_source_ref=AsyncMock(),
        delete_by_dataset_id=AsyncMock(),
    )


async def test_delete_data_routes_graph_native():
    dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()
    unified = _unified()

    with (
        patch.object(ddne_module, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))),
        patch.object(
            ddne_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
        ),
        patch.object(ddne_module, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(ddne_module, "is_graph_native_graph", AsyncMock(return_value=True)),
        patch.object(ddne_module, "delete_from_graph_and_vector", AsyncMock()) as legacy_delete,
    ):
        await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

    unified.delete_by_source_ref.assert_awaited_once_with(make_source_ref_key(dataset_id, data_id))
    legacy_delete.assert_not_called()  # returned before the ledger path


async def test_delete_dataset_routes_graph_native():
    dataset_id, user_id = uuid4(), uuid4()
    unified = _unified()

    with (
        patch.object(ddsne_module, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))),
        patch.object(
            ddsne_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
        ),
        patch.object(ddsne_module, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(ddsne_module, "is_graph_native_graph", AsyncMock(return_value=True)),
        patch.object(ddsne_module, "delete_from_graph_and_vector", AsyncMock()) as legacy_delete,
    ):
        await ddsne_module.delete_dataset_nodes_and_edges(dataset_id, user_id)

    unified.delete_by_dataset_id.assert_awaited_once_with(str(dataset_id))
    legacy_delete.assert_not_called()


async def test_delete_data_old_graph_uses_legacy():
    """Marker absent -> the unified graph-native delete is NOT called and the
    relational-ledger cleanup runs instead."""
    dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()
    unified = _unified(graph_native_supported=True)

    with (
        patch.object(ddne_module, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))),
        patch.object(
            ddne_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
        ),
        patch.object(ddne_module, "get_unified_engine", AsyncMock(return_value=unified)),
        # Old graph: marker absent.
        patch.object(ddne_module, "is_graph_native_graph", AsyncMock(return_value=False)),
        patch.object(ddne_module, "backend_access_control_enabled", lambda: False),
        patch.object(ddne_module, "get_global_data_related_nodes", AsyncMock(return_value=[])),
        patch.object(
            ddne_module, "get_shared_slugs_losing_dataset_anchor", AsyncMock(return_value=[])
        ),
        patch.object(ddne_module, "delete_data_related_nodes", AsyncMock()) as del_rel_nodes,
        patch.object(ddne_module, "delete_data_related_edges", AsyncMock()),
    ):
        await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

    unified.delete_by_source_ref.assert_not_called()
    del_rel_nodes.assert_awaited_once()  # legacy ledger cleanup ran

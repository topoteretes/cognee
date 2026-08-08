"""Routing seam tests for graph-provenance delete (COG-5522).

Prove that the public delete entry points send graph-provenance graphs through the
unified boundary (and old/unmarked graphs stay on the relational-ledger path).
These mock the unified engine + marker so they stay fast and backend-free; the
real end-to-end behavior is covered by the integration suite.
"""

import importlib
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# The package __init__ re-exports these functions under the same name as their
# submodule, so `import a.b.c as x` would bind the function, not the module.
# Pull the real module objects from sys.modules instead.
import cognee.modules.graph.methods.delete_data_nodes_and_edges  # noqa: F401
import cognee.modules.graph.methods.delete_dataset_nodes_and_edges  # noqa: F401
import cognee.modules.graph.methods.try_delete_data_by_graph_provenance  # noqa: F401
from cognee.infrastructure.databases.provenance import make_source_ref_key

ddne_module = sys.modules["cognee.modules.graph.methods.delete_data_nodes_and_edges"]
ddsne_module = sys.modules["cognee.modules.graph.methods.delete_dataset_nodes_and_edges"]
try_delete_module = sys.modules["cognee.modules.graph.methods.try_delete_data_by_graph_provenance"]
datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")

pytestmark = pytest.mark.asyncio


def _unified(graph_provenance_supported=True):
    return SimpleNamespace(
        supports_graph_provenance_delete=lambda: graph_provenance_supported,
        graph=object(),
        delete_by_source_ref=AsyncMock(),
        delete_by_dataset_id=AsyncMock(),
    )


@asynccontextmanager
async def _context():
    yield


async def test_try_delete_data_by_graph_provenance_deletes_marked_graph():
    dataset_id, data_id = uuid4(), uuid4()
    unified = _unified()

    with (
        patch.object(try_delete_module, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(try_delete_module, "stores_provenance_in_graph", AsyncMock(return_value=True)),
    ):
        handled = await try_delete_module.try_delete_data_by_graph_provenance(dataset_id, data_id)

    assert handled is True
    unified.delete_by_source_ref.assert_awaited_once_with(make_source_ref_key(dataset_id, data_id))


async def test_try_delete_data_by_graph_provenance_returns_false_when_unsupported():
    dataset_id, data_id = uuid4(), uuid4()
    unified = _unified(graph_provenance_supported=False)

    with (
        patch.object(try_delete_module, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(try_delete_module, "stores_provenance_in_graph", AsyncMock()) as marker,
    ):
        handled = await try_delete_module.try_delete_data_by_graph_provenance(dataset_id, data_id)

    assert handled is False
    marker.assert_not_called()
    unified.delete_by_source_ref.assert_not_called()


async def test_try_delete_data_by_graph_provenance_returns_false_when_unmarked():
    dataset_id, data_id = uuid4(), uuid4()
    unified = _unified()

    with (
        patch.object(try_delete_module, "get_unified_engine", AsyncMock(return_value=unified)),
        patch.object(
            try_delete_module, "stores_provenance_in_graph", AsyncMock(return_value=False)
        ),
    ):
        handled = await try_delete_module.try_delete_data_by_graph_provenance(dataset_id, data_id)

    assert handled is False
    unified.delete_by_source_ref.assert_not_called()


async def test_delete_data_routes_graph_provenance():
    dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()

    with (
        patch.object(ddne_module, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))),
        patch.object(
            ddne_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
        ),
        patch.object(
            ddne_module,
            "try_delete_data_by_graph_provenance",
            AsyncMock(return_value=True),
        ) as graph_delete,
        patch.object(ddne_module, "delete_from_graph_and_vector", AsyncMock()) as legacy_delete,
    ):
        await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

    graph_delete.assert_awaited_once_with(dataset_id, data_id)
    legacy_delete.assert_not_called()  # returned before the ledger path


async def test_delete_dataset_routes_graph_provenance():
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
        patch.object(ddsne_module, "stores_provenance_in_graph", AsyncMock(return_value=True)),
        patch.object(ddsne_module, "delete_from_graph_and_vector", AsyncMock()) as legacy_delete,
    ):
        await ddsne_module.delete_dataset_nodes_and_edges(dataset_id, user_id)

    unified.delete_by_dataset_id.assert_awaited_once_with(str(dataset_id))
    legacy_delete.assert_not_called()


async def test_delete_data_old_graph_uses_legacy():
    """Marker absent -> the unified graph-provenance delete is NOT called and the
    relational-ledger cleanup runs instead."""
    dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()

    with (
        patch.object(ddne_module, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))),
        patch.object(
            ddne_module,
            "get_authorized_dataset",
            AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
        ),
        patch.object(
            ddne_module,
            "try_delete_data_by_graph_provenance",
            AsyncMock(return_value=False),
        ) as graph_delete,
        patch.object(ddne_module, "backend_access_control_enabled", lambda: False),
        patch.object(ddne_module, "get_global_data_related_nodes", AsyncMock(return_value=[])),
        patch.object(
            ddne_module, "get_shared_slugs_losing_dataset_anchor", AsyncMock(return_value=[])
        ),
        patch.object(ddne_module, "delete_data_related_nodes", AsyncMock()) as del_rel_nodes,
        patch.object(ddne_module, "delete_data_related_edges", AsyncMock()),
    ):
        await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

    graph_delete.assert_awaited_once_with(dataset_id, data_id)
    del_rel_nodes.assert_awaited_once()  # legacy ledger cleanup ran


async def test_api_delete_data_uses_graph_provenance_when_ledger_has_no_nodes():
    dataset_id, data_id, user_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=user_id)
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, datasets=[SimpleNamespace(id=dataset_id)])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=[[data], [data]])),
        patch.object(
            datasets_module,
            "set_database_global_context_variables",
            lambda *_args, **_kwargs: _context(),
        ),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=False)),
        patch.object(
            datasets_module,
            "try_delete_data_by_graph_provenance",
            AsyncMock(return_value=True),
        ) as graph_delete,
        patch.object(datasets_module, "delete_data_nodes_and_edges", AsyncMock()) as ledger_delete,
        patch.object(datasets_module, "legacy_delete", AsyncMock()) as legacy_delete,
        patch.object(data_methods_module, "delete_data", AsyncMock()) as delete_data_row,
        patch.object(data_methods_module, "delete_dataset", AsyncMock()),
    ):
        result = await datasets_module.datasets.delete_data(dataset_id, data_id, user)

    assert result == {"status": "success"}
    graph_delete.assert_awaited_once_with(dataset_id, data_id)
    ledger_delete.assert_not_called()
    legacy_delete.assert_not_called()
    delete_data_row.assert_awaited_once_with(data, dataset_id)


async def test_api_delete_data_uses_legacy_when_no_ledger_nodes_and_unmarked_graph():
    dataset_id, data_id, user_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=user_id)
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, datasets=[SimpleNamespace(id=dataset_id)])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=[[data], [data]])),
        patch.object(
            datasets_module,
            "set_database_global_context_variables",
            lambda *_args, **_kwargs: _context(),
        ),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=False)),
        patch.object(
            datasets_module,
            "try_delete_data_by_graph_provenance",
            AsyncMock(return_value=False),
        ) as graph_delete,
        patch.object(datasets_module, "delete_data_nodes_and_edges", AsyncMock()) as ledger_delete,
        patch.object(datasets_module, "legacy_delete", AsyncMock()) as legacy_delete,
        patch.object(data_methods_module, "delete_data", AsyncMock()) as delete_data_row,
        patch.object(data_methods_module, "delete_dataset", AsyncMock()),
    ):
        result = await datasets_module.datasets.delete_data(dataset_id, data_id, user)

    assert result == {"status": "success"}
    graph_delete.assert_awaited_once_with(dataset_id, data_id)
    ledger_delete.assert_not_called()
    legacy_delete.assert_awaited_once_with(data, "soft")
    delete_data_row.assert_awaited_once_with(data, dataset_id)


async def test_api_delete_data_uses_ledger_delete_when_ledger_has_nodes():
    dataset_id, data_id, user_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=user_id)
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, datasets=[SimpleNamespace(id=dataset_id)])

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=[[data], [data]])),
        patch.object(
            datasets_module,
            "set_database_global_context_variables",
            lambda *_args, **_kwargs: _context(),
        ),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=True)),
        patch.object(
            datasets_module,
            "try_delete_data_by_graph_provenance",
            AsyncMock(),
        ) as graph_delete,
        patch.object(
            datasets_module,
            "delete_data_nodes_and_edges",
            AsyncMock(),
        ) as ledger_delete,
        patch.object(datasets_module, "legacy_delete", AsyncMock()) as legacy_delete,
        patch.object(data_methods_module, "delete_data", AsyncMock()) as delete_data_row,
        patch.object(data_methods_module, "delete_dataset", AsyncMock()),
    ):
        result = await datasets_module.datasets.delete_data(dataset_id, data_id, user)

    assert result == {"status": "success"}
    ledger_delete.assert_awaited_once_with(dataset_id, data_id, user.id)
    graph_delete.assert_not_called()
    legacy_delete.assert_not_called()
    delete_data_row.assert_awaited_once_with(data, dataset_id)

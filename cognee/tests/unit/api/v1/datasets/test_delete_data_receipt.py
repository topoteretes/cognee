"""``datasets.delete_data`` returns a verifiable deletion receipt.

The receipt keeps the historical ``{"status": "success"}`` key and adds what
was removed (``deleted_nodes`` / ``deleted_edges``), whether a Data row existed
(``data_record_found``), and — observed from a re-list after the delete, not
inferred — whether the Data row is still present (``data_remaining``). A caller
that needs proof of removal reads ``data_remaining is False`` instead of
listing the dataset itself.
"""

import importlib
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")
forget_module = importlib.import_module("cognee.api.v1.forget.forget")


@asynccontextmanager
async def _context(*_args, **_kwargs):
    yield


def _patches(dataset, listings, deleted_elements, delete_dataset=None):
    """Patch the delete path for the relational-ledger branch."""
    return (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=listings)),
        patch.object(datasets_module, "set_database_global_context_variables", _context),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=True)),
        patch.object(
            datasets_module,
            "delete_data_nodes_and_edges",
            AsyncMock(return_value=deleted_elements),
        ),
        patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=None)),
        patch.object(
            datasets_module, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()
        ),
        patch.object(data_methods_module, "delete_data", AsyncMock()),
        patch.object(data_methods_module, "delete_dataset", delete_dataset or AsyncMock()),
    )


async def _run(dataset_id, data_id, user, dataset, listings, deleted_elements, **kwargs):
    delete_dataset = kwargs.pop("delete_dataset", None)
    with ExitStack() as stack:
        for patcher in _patches(dataset, listings, deleted_elements, delete_dataset):
            stack.enter_context(patcher)
        return await datasets_module.datasets.delete_data(dataset_id, data_id, user, **kwargs)


@pytest.mark.asyncio
async def test_receipt_reports_removed_elements_and_verified_absence():
    dataset_id, data_id, owner_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, dataset_id=dataset_id)
    other = SimpleNamespace(id=uuid4(), dataset_id=dataset_id)
    deleted = DeletedGraphElements(node_ids={"n1", "n2", "n3"}, edge_ids={"e1", "e2"})

    # First listing finds the row; the post-delete re-list no longer shows it.
    result = await _run(dataset_id, data_id, user, dataset, [[data, other], [other]], deleted)

    assert result == {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": True,
        "deleted_nodes": 3,
        "deleted_edges": 2,
        "data_remaining": False,
        "dataset_deleted": False,
    }


@pytest.mark.asyncio
async def test_receipt_reports_data_remaining_when_relist_still_shows_the_row():
    """``data_remaining`` is observed from the re-list, never assumed."""
    dataset_id, data_id, owner_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, dataset_id=dataset_id)

    result = await _run(
        dataset_id, data_id, user, dataset, [[data], [data]], DeletedGraphElements()
    )

    assert result["status"] == "success"
    assert result["data_record_found"] is True
    assert result["data_remaining"] is True
    assert result["deleted_nodes"] == 0
    assert result["deleted_edges"] == 0


@pytest.mark.asyncio
async def test_receipt_marks_dataset_deleted_when_emptied_and_requested():
    dataset_id, data_id, owner_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, dataset_id=dataset_id)
    delete_dataset = AsyncMock()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        [[data], []],
        DeletedGraphElements(node_ids={"n1"}),
        delete_dataset_if_empty=True,
        delete_dataset=delete_dataset,
    )

    delete_dataset.assert_awaited_once_with(dataset)
    assert result["dataset_deleted"] is True
    assert result["data_remaining"] is False


@pytest.mark.asyncio
async def test_receipt_for_untracked_id_reports_no_data_record():
    """The custom-graph-model path (no Data row) still returns a full receipt."""
    dataset_id, data_id, owner_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    deleted = DeletedGraphElements(node_ids={"n1", "n2"}, edge_ids={"e1"})

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=[[], []])),
        patch.object(datasets_module, "set_database_global_context_variables", _context),
        patch.object(
            datasets_module, "delete_data_nodes_and_edges", AsyncMock(return_value=deleted)
        ),
        patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=None)),
        patch.object(
            datasets_module, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()
        ),
    ):
        result = await datasets_module.datasets.delete_data(dataset_id, data_id, user)

    assert result == {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": False,
        "deleted_nodes": 2,
        "deleted_edges": 1,
        "data_remaining": False,
        "dataset_deleted": False,
    }


@pytest.mark.asyncio
async def test_receipt_uses_resolved_legacy_id():
    """A legacy id resolves to the current row; the receipt names the resolved id."""
    dataset_id, legacy_id, resolved_id, owner_id = uuid4(), uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=resolved_id, dataset_id=dataset_id)

    with (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(side_effect=[[data], []])),
        patch.object(datasets_module, "set_database_global_context_variables", _context),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=True)),
        patch.object(
            datasets_module,
            "delete_data_nodes_and_edges",
            AsyncMock(return_value=DeletedGraphElements()),
        ),
        patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=resolved_id)),
        patch.object(
            datasets_module, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()
        ),
        patch.object(data_methods_module, "delete_data", AsyncMock()),
        patch.object(data_methods_module, "delete_dataset", AsyncMock()),
    ):
        result = await datasets_module.datasets.delete_data(dataset_id, legacy_id, user)

    assert result["data_id"] == str(resolved_id)
    assert result["data_record_found"] is True
    assert result["data_remaining"] is False


@pytest.mark.asyncio
async def test_forget_data_item_passes_the_receipt_through():
    """``forget(data_id=...)`` exposes the receipt fields alongside its own keys."""
    dataset_id, data_id = uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    receipt = {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": True,
        "deleted_nodes": 4,
        "deleted_edges": 3,
        "data_remaining": False,
        "dataset_deleted": False,
    }

    with (
        patch.object(forget_module, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch.object(datasets_module.datasets, "delete_data", AsyncMock(return_value=receipt)),
    ):
        result = await forget_module._forget_data_item(data_id, dataset_id, user)

    assert result == receipt


@pytest.mark.asyncio
async def test_forget_data_item_keeps_the_resolved_id_over_a_legacy_input():
    """A legacy id passed to ``forget`` resolves inside ``delete_data``; the
    receipt names the resolved id and ``forget`` must not overwrite it with the
    caller's legacy value."""
    dataset_id, legacy_id, resolved_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    receipt = {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(resolved_id),
        "data_record_found": True,
        "deleted_nodes": 1,
        "deleted_edges": 0,
        "data_remaining": False,
        "dataset_deleted": False,
    }

    with (
        patch.object(forget_module, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch.object(datasets_module.datasets, "delete_data", AsyncMock(return_value=receipt)),
    ):
        result = await forget_module._forget_data_item(legacy_id, dataset_id, user)

    assert result["data_id"] == str(resolved_id)
    assert result == receipt


def test_delete_route_answers_with_the_receipt_dto():
    """``DELETE /datasets/{id}/data/{id}`` declares ``DeleteDataReceiptDTO`` and
    serializes the receipt through it: every field present, camelCase on the wire,
    ids as strings."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.api.v1.datasets.dto import DeleteDataReceiptDTO
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router
    from cognee.modules.users.methods import get_authenticated_user

    dataset_id, data_id = uuid4(), uuid4()
    receipt = {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": True,
        "deleted_nodes": 4,
        "deleted_edges": 3,
        "data_remaining": False,
        "dataset_deleted": False,
    }

    router = get_datasets_router()
    route = next(
        r
        for r in router.routes
        if getattr(r, "path", "") == "/{dataset_id}/data/{data_id}"
        and "DELETE" in (getattr(r, "methods", None) or set())
    )
    assert route.response_model is DeleteDataReceiptDTO

    app = FastAPI()
    app.include_router(router, prefix="/datasets")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())

    router_module = importlib.import_module("cognee.api.v1.datasets.routers.get_datasets_router")

    with (
        patch.object(
            router_module.datasets, "delete_data", AsyncMock(return_value=receipt)
        ) as delete_data,
        patch.object(router_module, "send_telemetry"),
    ):
        response = TestClient(app).delete(f"/datasets/{dataset_id}/data/{data_id}")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "datasetId": str(dataset_id),
        "dataId": str(data_id),
        "dataRecordFound": True,
        "deletedNodes": 4,
        "deletedEdges": 3,
        "dataRemaining": False,
        "datasetDeleted": False,
    }
    called_dataset_id, called_data_id, _user = delete_data.await_args.args
    assert (called_dataset_id, called_data_id) == (dataset_id, data_id)

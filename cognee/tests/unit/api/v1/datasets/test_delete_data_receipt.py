"""``datasets.delete_data`` returns a verifiable deletion receipt.

A returned receipt means the delete succeeded: an id with nothing to remove
raises ``DocumentNotFoundError`` (404) and a Data row that survives the delete
raises ``DataDeletionIncompleteError`` (500), so ``status`` can never contradict
the outcome. The receipt keeps the historical ``{"status": "success"}`` key and
adds what was removed (``deleted_nodes`` / ``deleted_edges``), whether a Data row
existed (``data_record_found``) and ``data_remaining`` — ``False`` observed by a
single-row existence check after the delete, ``None`` when no row existed.
``DeleteDataReceiptDTO`` is the one definition of the shape.
"""

import importlib
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.datasets.dto import DeleteDataReceiptDTO
from cognee.api.v1.exceptions import DataDeletionIncompleteError
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")
search_operations_module = importlib.import_module("cognee.modules.search.operations")
forget_module = importlib.import_module("cognee.api.v1.forget.forget")


@asynccontextmanager
async def _context(*_args, **_kwargs):
    yield


def _patches(
    dataset,
    data,
    deleted_elements,
    *,
    row_remaining=False,
    dataset_has_data=True,
    delete_dataset=None,
    has_dataset_data=None,
    session_invalidate=None,
    delete_history=None,
    retired_decisions=frozenset(),
):
    """Patch the delete path. ``data`` is the row ``get_data`` resolves (None for
    the custom-graph-model path); ``row_remaining`` is what the post-delete
    existence check observes."""
    return (
        patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(data_methods_module, "get_data", AsyncMock(return_value=data)),
        patch.object(datasets_module, "set_database_global_context_variables", _context),
        patch.object(datasets_module, "has_data_related_nodes", AsyncMock(return_value=True)),
        patch.object(
            datasets_module,
            "delete_data_nodes_and_edges",
            AsyncMock(return_value=deleted_elements),
        ),
        patch.object(
            datasets_module,
            "retire_forgotten_decisions",
            AsyncMock(return_value=set(retired_decisions)),
        ),
        patch.object(
            datasets_module,
            "_invalidate_sessions_for_deleted_data_nonfatal",
            session_invalidate or AsyncMock(),
        ),
        patch.object(data_methods_module, "delete_data", AsyncMock()),
        patch.object(datasets_module, "has_data", AsyncMock(return_value=row_remaining)),
        patch.object(
            datasets_module,
            "has_dataset_data",
            has_dataset_data or AsyncMock(return_value=dataset_has_data),
        ),
        patch.object(data_methods_module, "delete_dataset", delete_dataset or AsyncMock()),
        patch.object(
            search_operations_module,
            "delete_dataset_search_history",
            delete_history or AsyncMock(),
        ),
    )


async def _run(dataset_id, data_id, user, dataset, data, deleted_elements, **kwargs):
    call_kwargs = {key: kwargs.pop(key) for key in ("delete_dataset_if_empty",) if key in kwargs}
    with ExitStack() as stack:
        for patcher in _patches(dataset, data, deleted_elements, **kwargs):
            stack.enter_context(patcher)
        return await datasets_module.datasets.delete_data(dataset_id, data_id, user, **call_kwargs)


def _fixture():
    dataset_id, data_id, owner_id = uuid4(), uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=dataset_id, owner_id=owner_id)
    data = SimpleNamespace(id=data_id, dataset_id=dataset_id)
    return dataset_id, data_id, user, dataset, data


@pytest.mark.asyncio
async def test_receipt_reports_removed_elements_and_verified_absence():
    dataset_id, data_id, user, dataset, data = _fixture()
    deleted = DeletedGraphElements(node_ids={"n1", "n2", "n3"}, edge_ids={"e1", "e2"})

    result = await _run(dataset_id, data_id, user, dataset, data, deleted)

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
async def test_a_retired_decision_is_invalidated_in_sessions_but_not_counted_in_the_receipt():
    dataset_id, data_id, user, dataset, data = _fixture()
    deleted = DeletedGraphElements(node_ids={"n1"}, edge_ids={"e1"})
    invalidate = AsyncMock()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        data,
        deleted,
        session_invalidate=invalidate,
        retired_decisions={"conflict"},
    )

    invalidated = invalidate.await_args.args[1]
    assert invalidated.node_ids == {"n1", "conflict"} and invalidated.edge_ids == {"e1"}
    assert result["deleted_nodes"] == 1


@pytest.mark.asyncio
async def test_delete_fails_when_the_row_survives_instead_of_reporting_success():
    """A Data row still present after the delete is a failed delete: raise, never
    answer ``status: success`` with ``data_remaining: True``."""
    dataset_id, data_id, user, dataset, data = _fixture()

    with pytest.raises(DataDeletionIncompleteError) as excinfo:
        await _run(
            dataset_id, data_id, user, dataset, data, DeletedGraphElements(), row_remaining=True
        )

    assert excinfo.value.status_code == 500
    assert str(data_id) in excinfo.value.message


@pytest.mark.asyncio
async def test_receipt_marks_dataset_deleted_when_emptied_and_requested():
    """The emptied dataset goes as in ``empty_dataset``: its search history first,
    then the dataset."""
    dataset_id, data_id, user, dataset, data = _fixture()
    order = []
    delete_dataset = AsyncMock(side_effect=lambda *_: order.append("dataset"))
    delete_history = AsyncMock(side_effect=lambda *_: order.append("history"))

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        data,
        DeletedGraphElements(node_ids={"n1"}),
        delete_dataset_if_empty=True,
        dataset_has_data=False,
        delete_dataset=delete_dataset,
        delete_history=delete_history,
    )

    delete_dataset.assert_awaited_once_with(dataset)
    delete_history.assert_awaited_once_with(dataset_id)
    assert order == ["history", "dataset"]
    assert result["dataset_deleted"] is True
    assert result["data_remaining"] is False


@pytest.mark.asyncio
async def test_emptiness_is_a_count_query_and_only_when_requested():
    """The receipt never lists the dataset: emptiness is ``has_dataset_data`` and it is
    not even asked when the caller did not request the dataset's removal."""
    dataset_id, data_id, user, dataset, data = _fixture()
    has_dataset_data = AsyncMock(return_value=True)
    delete_dataset = AsyncMock()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        data,
        DeletedGraphElements(),
        has_dataset_data=has_dataset_data,
        delete_dataset=delete_dataset,
    )
    has_dataset_data.assert_not_awaited()
    assert result["dataset_deleted"] is False

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        data,
        DeletedGraphElements(),
        delete_dataset_if_empty=True,
        has_dataset_data=has_dataset_data,
        delete_dataset=delete_dataset,
    )
    has_dataset_data.assert_awaited_once_with(dataset_id)
    delete_dataset.assert_not_awaited()
    assert result["dataset_deleted"] is False


@pytest.mark.asyncio
async def test_receipt_for_untracked_id_reports_no_data_record_and_nothing_observed():
    """The custom-graph-model path (no Data row) returns a receipt whose
    ``data_remaining`` is ``None``: there was no row to verify the absence of."""
    dataset_id, data_id, user, dataset, _data = _fixture()
    deleted = DeletedGraphElements(node_ids={"n1", "n2"}, edge_ids={"e1"}, owned=True)

    result = await _run(dataset_id, data_id, user, dataset, None, deleted)

    assert result == {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": False,
        "deleted_nodes": 2,
        "deleted_edges": 1,
        "data_remaining": None,
        "dataset_deleted": False,
    }


@pytest.mark.asyncio
async def test_untracked_id_owning_only_shared_elements_reports_success():
    """An id whose graph elements are all shared with other data hard-deletes
    nothing, but its ownership was removed: ``success`` with zero counts, and no
    session invalidation since no element left the graph."""
    dataset_id, data_id, user, dataset, _data = _fixture()
    session_invalidate = AsyncMock()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        None,
        DeletedGraphElements(owned=True),
        session_invalidate=session_invalidate,
    )

    assert result["status"] == "success"
    assert result["deleted_nodes"] == 0
    assert result["deleted_edges"] == 0
    session_invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_untracked_id_that_owns_nothing_reports_not_found():
    """A mistyped id, another dataset's id, or an already-deleted one: no Data row
    and no graph element to remove. The receipt says so (``status: not_found``,
    zero counts) instead of ``success``; deleting stays idempotent, so it does not
    raise, and ``delete_dataset_if_empty`` is still honoured — re-deleting the last
    item can still drop the now-empty dataset."""
    dataset_id, data_id, user, dataset, _data = _fixture()
    session_invalidate = AsyncMock()
    delete_dataset = AsyncMock()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        None,
        DeletedGraphElements(),
        session_invalidate=session_invalidate,
        delete_dataset=delete_dataset,
    )

    assert result == {
        "status": "not_found",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": False,
        "deleted_nodes": 0,
        "deleted_edges": 0,
        "data_remaining": None,
        "dataset_deleted": False,
    }
    session_invalidate.assert_not_awaited()
    delete_dataset.assert_not_awaited()

    result = await _run(
        dataset_id,
        data_id,
        user,
        dataset,
        None,
        DeletedGraphElements(),
        delete_dataset_if_empty=True,
        dataset_has_data=False,
        delete_dataset=delete_dataset,
    )

    assert result["status"] == "not_found"
    assert result["dataset_deleted"] is True
    delete_dataset.assert_awaited_once_with(dataset)


@pytest.mark.asyncio
async def test_receipt_uses_resolved_legacy_id():
    """A legacy id resolves (inside ``get_data``) to the current row; the receipt
    names the resolved id and the row delete targets that row."""
    dataset_id, legacy_id, user, dataset, _data = _fixture()
    resolved = SimpleNamespace(id=uuid4(), dataset_id=dataset_id)

    with ExitStack() as stack:
        for patcher in _patches(dataset, resolved, DeletedGraphElements()):
            stack.enter_context(patcher)
        delete_row = stack.enter_context(
            patch.object(data_methods_module, "delete_data", AsyncMock())
        )
        get_data = stack.enter_context(
            patch.object(data_methods_module, "get_data", AsyncMock(return_value=resolved))
        )
        result = await datasets_module.datasets.delete_data(dataset_id, legacy_id, user)

    get_data.assert_awaited_once_with(user.id, legacy_id, dataset_id, verify_owner=False)
    delete_row.assert_awaited_once_with([resolved], dataset_id)
    assert result["data_id"] == str(resolved.id)
    assert result["data_record_found"] is True
    assert result["data_remaining"] is False


def test_receipt_shape_is_the_dto():
    """``_delete_data_receipt`` is derived from ``DeleteDataReceiptDTO``: same keys,
    snake_case, ids as strings, and the dict re-validates through the model."""
    dataset_id, data_id = uuid4(), uuid4()

    receipt = datasets_module._delete_data_receipt(
        dataset_id=dataset_id,
        data_id=data_id,
        data_record_found=False,
        deleted_elements=DeletedGraphElements(node_ids={"n1"}),
        data_remaining=None,
        dataset_deleted=False,
    )

    assert list(receipt) == list(DeleteDataReceiptDTO.model_fields)
    assert receipt["dataset_id"] == str(dataset_id)
    assert receipt["data_remaining"] is None
    parsed = DeleteDataReceiptDTO.model_validate(receipt)
    assert parsed.data_id == data_id
    assert parsed.model_dump(mode="json") == receipt


@pytest.mark.asyncio
async def test_forget_data_item_returns_the_receipt_unchanged():
    """``forget(data_id=...)`` returns what ``delete_data`` produced, so the resolved
    ``data_id`` is never overwritten with the caller's (legacy) value. The one key
    it adds is ``search_history_removed`` (always False for one item)."""
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
        patch.object(
            datasets_module.datasets, "delete_data", AsyncMock(return_value=receipt)
        ) as delete_data,
    ):
        result = await forget_module._forget_data_item(legacy_id, dataset_id, user)

    assert result == {**receipt, "search_history_removed": False}
    assert result["data_id"] == str(resolved_id)
    delete_data.assert_awaited_once_with(
        dataset_id=dataset_id, data_id=legacy_id, user=user, delete_dataset_if_empty=False
    )


@pytest.mark.asyncio
async def test_forget_data_item_logs_not_found_instead_of_deleted():
    """A ``not_found`` receipt is never logged as a deletion."""
    dataset_id, data_id = uuid4(), uuid4()
    user = SimpleNamespace(id=uuid4())
    receipt = {"status": "not_found", "dataset_id": str(dataset_id), "data_id": str(data_id)}

    with (
        patch.object(forget_module, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch.object(datasets_module.datasets, "delete_data", AsyncMock(return_value=receipt)),
        patch.object(forget_module, "logger") as logger,
    ):
        result = await forget_module._forget_data_item(data_id, dataset_id, user)

    assert result == {**receipt, "search_history_removed": False}
    logger.info.assert_called_once()
    message = logger.info.call_args.args[0]
    assert "nothing to delete" in message and "not found" in message
    assert "deleted" not in message


def _receipt_dict(dataset_id, data_id, status="success"):
    return {
        "status": status,
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": True,
        "deleted_nodes": 2,
        "deleted_edges": 1,
        "data_remaining": False,
        "dataset_deleted": False,
    }


def test_receipt_casing_is_pinned_across_the_three_delete_routes():
    """One receipt, one wire shape per route family. ``DELETE /datasets/{id}/data/{id}``
    and the deprecated ``DELETE /delete`` both declare ``DeleteDataReceiptDTO`` and send
    identical camelCase bodies. ``POST /forget`` answers a union of shapes and stays
    snake_case: its body for a data_id delete is the DTO's fields in snake_case, as its
    docstring says."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router
    from cognee.api.v1.delete.routers.get_delete_router import get_delete_router
    from cognee.api.v1.forget.routers.get_forget_router import get_forget_router
    from cognee.modules.users.methods import get_authenticated_user

    dataset_id, data_id = uuid4(), uuid4()
    receipt = _receipt_dict(dataset_id, data_id)
    camel = {
        "status": "success",
        "datasetId": str(dataset_id),
        "dataId": str(data_id),
        "dataRecordFound": True,
        "deletedNodes": 2,
        "deletedEdges": 1,
        "dataRemaining": False,
        "datasetDeleted": False,
    }

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/datasets")
    app.include_router(get_delete_router(), prefix="/delete")
    app.include_router(get_forget_router(), prefix="/forget")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    delete_router_module = importlib.import_module("cognee.api.v1.delete.routers.get_delete_router")
    forget_router_module = importlib.import_module("cognee.api.v1.forget.routers.get_forget_router")
    forget_pkg = importlib.import_module("cognee.api.v1.forget")

    with (
        patch.object(datasets_module.datasets, "delete_data", AsyncMock(return_value=receipt)),
        patch.object(forget_pkg, "forget", AsyncMock(return_value=dict(receipt))),
        patch.object(datasets_router_module, "send_telemetry"),
        patch.object(delete_router_module, "send_telemetry"),
        patch.object(forget_router_module, "send_telemetry"),
    ):
        client = TestClient(app)
        via_datasets = client.delete(f"/datasets/{dataset_id}/data/{data_id}")
        via_delete = client.delete(f"/delete?data_id={data_id}&dataset_id={dataset_id}")
        via_forget = client.post(
            "/forget", json={"datasetId": str(dataset_id), "dataId": str(data_id)}
        )

    assert via_datasets.status_code == via_delete.status_code == via_forget.status_code == 200
    assert via_datasets.json() == camel
    assert via_delete.json() == camel
    assert via_forget.json() == receipt
    assert set(via_forget.json()) == set(DeleteDataReceiptDTO.model_fields)


def test_delete_route_answers_with_the_receipt_dto():
    """``DELETE /datasets/{id}/data/{id}`` declares ``DeleteDataReceiptDTO`` and
    serializes the receipt through it: every field present, camelCase on the wire,
    ids as strings, ``dataRemaining`` nullable."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router
    from cognee.modules.users.methods import get_authenticated_user

    dataset_id, data_id = uuid4(), uuid4()
    receipt = {
        "status": "success",
        "dataset_id": str(dataset_id),
        "data_id": str(data_id),
        "data_record_found": False,
        "deleted_nodes": 4,
        "deleted_edges": 3,
        "data_remaining": None,
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
        "dataRecordFound": False,
        "deletedNodes": 4,
        "deletedEdges": 3,
        "dataRemaining": None,
        "datasetDeleted": False,
    }
    called_dataset_id, called_data_id, _user = delete_data.await_args.args
    assert (called_dataset_id, called_data_id) == (dataset_id, data_id)

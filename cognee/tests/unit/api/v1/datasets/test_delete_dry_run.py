"""dry_run previews for the datasets deletion API (issue #5672).

``datasets.delete_data``, ``datasets.empty_dataset`` and ``datasets.delete_all``
accept ``dry_run=True`` to report the scope a real run would remove without
changing anything: no delete lock, no graph/vector/relational writes, no
operation record, no session invalidation. The default ``dry_run=False`` keeps
the pre-existing behavior byte-for-byte.
"""

import importlib
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")

USER = SimpleNamespace(id=uuid4())
DATASET_ID = uuid4()
DATA_ID = uuid4()


def _dataset_stub(dataset_id=DATASET_ID):
    return SimpleNamespace(id=dataset_id, owner_id=USER.id, name="test_dataset")


def _data_stub(data_id=DATA_ID):
    return SimpleNamespace(id=data_id, name="research_notes.pdf", dataset_id=DATASET_ID)


class _NoopContextManager:
    """Stand-in for set_database_global_context_variables."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@contextmanager
def _patch_deletion_surface(*, dataset_data=None, delete_dataset_result=None):
    """Patch every store mutation and write the real delete paths can reach.

    Dry-run tests assert none of these fire; the dry_run=False regression
    tests assert exactly the ones the pre-existing behavior fires.

    Usable as ``with _patch_deletion_surface(...) as mocks:``.
    """
    mocks = {
        "delete_dataset_nodes_and_edges": AsyncMock(return_value=DeletedGraphElements()),
        "delete_data_nodes_and_edges": AsyncMock(return_value=DeletedGraphElements()),
        "try_delete_data_by_graph_provenance": AsyncMock(return_value=None),
        "legacy_delete": AsyncMock(return_value=DeletedGraphElements()),
        "has_data_related_nodes": AsyncMock(return_value=True),
        "record_operation": MagicMock(),
        "delete_data_method": AsyncMock(return_value=True),
        "delete_dataset_method": AsyncMock(return_value=delete_dataset_result),
        "invalidate_for_dataset": AsyncMock(),
        "invalidate_for_deleted_data": AsyncMock(),
    }

    with ExitStack() as stack:
        for entry in [
            patch.object(
                datasets_module,
                "get_authorized_dataset",
                # Reflect the requested dataset id back, like the real lookup.
                AsyncMock(
                    side_effect=lambda user, dataset_id, permission: _dataset_stub(dataset_id)
                ),
            ),
            patch.object(
                datasets_module, "set_database_global_context_variables", _NoopContextManager
            ),
            patch.object(
                datasets_module, "get_dataset_data", AsyncMock(return_value=dataset_data or [])
            ),
            patch.object(
                datasets_module,
                "delete_dataset_nodes_and_edges",
                mocks["delete_dataset_nodes_and_edges"],
            ),
            patch.object(
                datasets_module,
                "delete_data_nodes_and_edges",
                mocks["delete_data_nodes_and_edges"],
            ),
            patch.object(
                datasets_module,
                "try_delete_data_by_graph_provenance",
                mocks["try_delete_data_by_graph_provenance"],
            ),
            patch.object(datasets_module, "legacy_delete", mocks["legacy_delete"]),
            patch.object(
                datasets_module, "has_data_related_nodes", mocks["has_data_related_nodes"]
            ),
            patch.object(datasets_module, "record_operation", mocks["record_operation"]),
            patch.object(data_methods_module, "delete_data", mocks["delete_data_method"]),
            patch.object(data_methods_module, "delete_dataset", mocks["delete_dataset_method"]),
            patch.object(data_methods_module, "resolve_data_id", AsyncMock(return_value=None)),
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.invalidate_sessions_for_dataset",
                mocks["invalidate_for_dataset"],
            ),
            patch(
                "cognee.modules.session_lifecycle.invalidate_sessions.invalidate_sessions_for_deleted_data",
                mocks["invalidate_for_deleted_data"],
            ),
        ]:
            stack.enter_context(entry)
        yield mocks


@pytest.mark.asyncio
async def test_delete_data_dry_run_deletes_nothing():
    """dry_run=True on a tracked data item touches no store and no session."""
    with _patch_deletion_surface(dataset_data=[_data_stub()]) as mocks:
        report = await datasets_module.datasets.delete_data(DATASET_ID, DATA_ID, USER, dry_run=True)

    mocks["delete_data_nodes_and_edges"].assert_not_called()
    mocks["try_delete_data_by_graph_provenance"].assert_not_called()
    mocks["legacy_delete"].assert_not_called()
    mocks["has_data_related_nodes"].assert_not_called()
    mocks["delete_data_method"].assert_not_called()
    mocks["delete_dataset_method"].assert_not_called()
    mocks["invalidate_for_dataset"].assert_not_called()
    mocks["invalidate_for_deleted_data"].assert_not_called()
    mocks["record_operation"].assert_not_called()

    assert report == {
        "status": "dry_run",
        "dry_run": True,
        "operation": "delete_data",
        "dataset": {"id": str(DATASET_ID), "name": "test_dataset"},
        "data": {"id": str(DATA_ID), "name": "research_notes.pdf"},
        "tracked_data_row": True,
        # The stub dataset holds exactly this one item.
        "dataset_empty_after_delete": True,
        "would_delete_dataset": False,
    }


@pytest.mark.asyncio
async def test_delete_data_dry_run_reports_dataset_cascade():
    """delete_dataset_if_empty is reflected in the preview for the last item."""
    with _patch_deletion_surface(dataset_data=[_data_stub()]):
        report = await datasets_module.datasets.delete_data(
            DATASET_ID,
            DATA_ID,
            USER,
            delete_dataset_if_empty=True,
            dry_run=True,
        )

    assert report["dataset_empty_after_delete"] is True
    assert report["would_delete_dataset"] is True


@pytest.mark.asyncio
async def test_delete_data_dry_run_with_untracked_data_id():
    """A data id with no Data row previews the custom-graph-model deletion."""
    with _patch_deletion_surface(dataset_data=[_data_stub()]) as mocks:
        report = await datasets_module.datasets.delete_data(DATASET_ID, uuid4(), USER, dry_run=True)

    assert report["data"] is None
    assert report["tracked_data_row"] is False
    assert report["dataset"] == {"id": str(DATASET_ID), "name": "test_dataset"}
    mocks["delete_data_nodes_and_edges"].assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_on_missing_dataset_raises():
    """Preview permission checks match the real path: unknown dataset raises."""
    with patch.object(datasets_module, "get_authorized_dataset", AsyncMock(return_value=None)):
        with pytest.raises(UnauthorizedDataAccessError):
            await datasets_module.datasets.delete_data(DATASET_ID, DATA_ID, USER, dry_run=True)

        with pytest.raises(UnauthorizedDataAccessError):
            await datasets_module.datasets.empty_dataset(DATASET_ID, USER, dry_run=True)


@pytest.mark.asyncio
async def test_empty_dataset_dry_run_deletes_nothing():
    """dry_run=True lists the dataset's data items and removes nothing."""
    items = [_data_stub(), _data_stub(uuid4())]

    with _patch_deletion_surface(dataset_data=items) as mocks:
        report = await datasets_module.datasets.empty_dataset(DATASET_ID, USER, dry_run=True)

    mocks["delete_dataset_nodes_and_edges"].assert_not_called()
    mocks["delete_dataset_method"].assert_not_called()
    mocks["invalidate_for_dataset"].assert_not_called()
    mocks["invalidate_for_deleted_data"].assert_not_called()

    assert report == {
        "status": "dry_run",
        "dry_run": True,
        "operation": "empty_dataset",
        "dataset": {"id": str(DATASET_ID), "name": "test_dataset"},
        "data_items": [{"id": str(item.id), "name": item.name} for item in items],
        "data_count": 2,
    }


@pytest.mark.asyncio
async def test_delete_all_dry_run_deletes_nothing():
    """dry_run=True aggregates a per-dataset preview and removes nothing."""
    second_dataset_id = uuid4()
    datasets_to_authorize = [_dataset_stub(DATASET_ID), _dataset_stub(second_dataset_id)]
    items = [_data_stub(), _data_stub(uuid4())]

    with (
        _patch_deletion_surface(dataset_data=items) as mocks,
        patch.object(
            datasets_module,
            "get_authorized_existing_datasets",
            AsyncMock(return_value=datasets_to_authorize),
        ),
    ):
        report = await datasets_module.datasets.delete_all(USER, dry_run=True)

    mocks["delete_dataset_nodes_and_edges"].assert_not_called()
    mocks["delete_dataset_method"].assert_not_called()

    assert report["status"] == "dry_run"
    assert report["dry_run"] is True
    assert report["operation"] == "delete_all"
    assert report["dataset_count"] == 2
    # Both previews enumerated the same two stub items.
    assert report["data_count"] == 4
    assert [entry["dataset"]["id"] for entry in report["datasets"]] == [
        str(DATASET_ID),
        str(second_dataset_id),
    ]
    assert all(entry["operation"] == "empty_dataset" for entry in report["datasets"])


@pytest.mark.asyncio
async def test_delete_data_default_still_deletes():
    """Regression: the omitted dry_run flag keeps the real deletion path."""
    with _patch_deletion_surface(dataset_data=[_data_stub()]) as mocks:
        result = await datasets_module.datasets.delete_data(DATASET_ID, DATA_ID, USER)

    assert result == {"status": "success"}
    mocks["has_data_related_nodes"].assert_awaited_once()
    mocks["delete_data_nodes_and_edges"].assert_awaited_once()
    mocks["delete_data_method"].assert_awaited_once()
    mocks["invalidate_for_deleted_data"].assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_dataset_default_still_deletes():
    """Regression: the omitted dry_run flag keeps the real deletion path."""
    with _patch_deletion_surface(delete_dataset_result={"status": "success"}) as mocks:
        result = await datasets_module.datasets.empty_dataset(DATASET_ID, USER)

    assert result == {"status": "success"}
    mocks["delete_dataset_nodes_and_edges"].assert_awaited_once()
    mocks["delete_dataset_method"].assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_all_default_still_deletes_every_dataset():
    """Regression: delete_all without the flag empties each authorized dataset."""
    datasets_to_authorize = [_dataset_stub(DATASET_ID), _dataset_stub(uuid4())]

    with (
        patch.object(
            datasets_module,
            "get_authorized_existing_datasets",
            AsyncMock(return_value=datasets_to_authorize),
        ),
        patch.object(
            datasets_module.datasets, "empty_dataset", new_callable=AsyncMock
        ) as empty_mock,
    ):
        await datasets_module.datasets.delete_all(USER)

    assert empty_mock.await_count == 2
    assert [call.args for call in empty_mock.await_args_list] == [
        (dataset.id, USER) for dataset in datasets_to_authorize
    ]

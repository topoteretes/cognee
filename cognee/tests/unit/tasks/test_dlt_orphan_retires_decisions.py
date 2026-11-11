"""A connector row removed upstream retires the user decisions its facts backed."""

import importlib
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import held_datasets

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")


@contextmanager
def _patches(dataset, orphan, retire, delete, invalidate):
    @asynccontextmanager
    async def context(*args):
        yield

    with (
        patch(
            "cognee.modules.data.methods.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch.object(
            importlib.import_module("cognee.modules.data.methods.get_dataset_data"),
            "get_dataset_data",
            new=AsyncMock(return_value=[orphan]),
        ),
        patch("cognee.context_global_variables.set_database_global_context_variables", context),
        patch.object(
            importlib.import_module("cognee.modules.graph.methods.delete_data_nodes_and_edges"),
            "delete_data_nodes_and_edges",
            new=AsyncMock(return_value=SimpleNamespace(node_ids={"n1"}, edge_ids={"e1"})),
        ),
        patch("cognee.modules.conflicts.methods.retire_forgotten_decisions", new=retire),
        patch.object(
            importlib.import_module("cognee.modules.data.methods.delete_data"),
            "delete_data",
            new=delete,
        ),
        patch.object(
            importlib.import_module("cognee.modules.session_lifecycle.invalidate_sessions"),
            "invalidate_sessions_for_deleted_data",
            new=invalidate,
        ),
    ):
        yield


def _fixture():
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    user = SimpleNamespace(id=uuid4())
    orphan = SimpleNamespace(
        id=uuid4(), system_metadata={"source": "notion", "table_name": "pages"}
    )
    return dataset, user, orphan


async def _cleanup(user):
    await resolve._delete_dlt_orphans(
        "dataset", user, set(), sources=("notion",), document_scopes={("notion", "pages")}
    )


@pytest.mark.asyncio
async def test_orphan_cleanup_retires_decisions_under_the_dataset_lock():
    dataset, user, orphan = _fixture()

    async def retire(retire_dataset, retire_user):
        assert dataset.id in held_datasets.get()
        assert (retire_dataset, retire_user) == (dataset, user)
        return {"conflict"}

    delete, invalidate = AsyncMock(), AsyncMock()
    with _patches(dataset, orphan, AsyncMock(side_effect=retire), delete, invalidate):
        await _cleanup(user)

    delete.assert_awaited_once_with([orphan], dataset.id)
    assert [call.args[1:] for call in invalidate.await_args_list] == [
        ({"n1"}, {"e1"}),
        ({"conflict"}, set()),
    ]


@pytest.mark.asyncio
async def test_a_failed_retirement_keeps_the_row_for_the_next_ingest():
    dataset, user, orphan = _fixture()
    delete = AsyncMock()
    retire = AsyncMock(side_effect=RuntimeError("graph read failed"))
    with _patches(dataset, orphan, retire, delete, AsyncMock()):
        await _cleanup(user)

    delete.assert_not_awaited()


def _ledger(tombstone):
    return patch("cognee.modules.provenance.tombstones.tombstone_data_items", new=tombstone)


@pytest.mark.asyncio
async def test_orphan_cleanup_tombstones_the_rows_it_removed_after_they_are_gone():
    dataset, user, orphan = _fixture()
    order: list[str] = []

    async def delete(rows, dataset_id):
        order.append("rows")

    async def tombstone(dataset_id, data_ids, *, user):
        assert dataset.id in held_datasets.get()
        order.append("tombstone")

    tombstone = AsyncMock(side_effect=tombstone)
    with (
        _patches(dataset, orphan, AsyncMock(return_value=set()), delete, AsyncMock()),
        _ledger(tombstone),
    ):
        await _cleanup(user)

    assert order == ["rows", "tombstone"]
    tombstone.assert_awaited_once_with(dataset.id, [orphan.id], user=user)


@pytest.mark.asyncio
async def test_orphan_cleanup_tombstones_nothing_when_the_row_delete_failed():
    dataset, user, orphan = _fixture()
    delete = AsyncMock(side_effect=RuntimeError("relational delete failed"))
    tombstone = AsyncMock()
    with (
        _patches(dataset, orphan, AsyncMock(return_value=set()), delete, AsyncMock()),
        _ledger(tombstone),
    ):
        await _cleanup(user)

    tombstone.assert_awaited_once_with(dataset.id, [], user=user)


@pytest.mark.asyncio
async def test_orphan_cleanup_raises_when_the_tombstones_cannot_be_written():
    dataset, user, orphan = _fixture()
    tombstone = AsyncMock(side_effect=RuntimeError("ledger down"))
    with (
        _patches(dataset, orphan, AsyncMock(return_value=set()), AsyncMock(), AsyncMock()),
        _ledger(tombstone),
        pytest.raises(RuntimeError, match="ledger down"),
    ):
        await _cleanup(user)

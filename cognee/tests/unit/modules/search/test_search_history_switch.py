"""``COGNEE_LOG_SEARCH_HISTORY`` and the forget paths that purge history.

The flag is read on every call, so it can be flipped after import; when it is
off nothing is written, not even the query row. The delete functions ignore
the flag: rows written while logging was on must still be removable.
"""

import importlib
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.search.models.Query import Query
from cognee.modules.search.models.Result import Result
from cognee.modules.search.operations import search_history_enabled

log_query_mod = importlib.import_module("cognee.modules.search.operations.log_query")
log_result_mod = importlib.import_module("cognee.modules.search.operations.log_result")
log_search_history_mod = importlib.import_module(
    "cognee.modules.search.operations.log_search_history"
)
delete_history_mod = importlib.import_module(
    "cognee.modules.search.operations.delete_search_history"
)
get_history_mod = importlib.import_module("cognee.modules.search.operations.get_history")
forget_mod = importlib.import_module("cognee.api.v1.forget.forget")
# The package re-exports the ``datasets`` class under the submodule's name, so a
# dotted ``patch("cognee.api.v1.datasets.datasets.datasets")`` resolves to the
# class on Python 3.10 (``mock._importer`` walks attributes, not modules).
# Patch the module object instead, which works on every version.
datasets_mod = importlib.import_module("cognee.api.v1.datasets.datasets")

COUNTS = {"queries_deleted": 1, "results_deleted": 1}


@pytest_asyncio.fixture
async def history_engine(tmp_path, monkeypatch):
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="history_switch.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all, tables=[Query.__table__, Result.__table__]
        )

    for module in (log_query_mod, log_result_mod, delete_history_mod, get_history_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _row_counts(engine) -> tuple[int, int]:
    async with engine.get_async_session() as session:
        queries = await session.scalar(select(func.count()).select_from(Query))
        results = await session.scalar(select(func.count()).select_from(Result))
    return queries, results


@pytest.mark.parametrize(
    "value, expected",
    [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("no", False)],
)
def test_flag_is_read_at_call_time(monkeypatch, value, expected):
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", value)
    assert search_history_enabled() is expected


def test_flag_defaults_to_on(monkeypatch):
    monkeypatch.delenv("COGNEE_LOG_SEARCH_HISTORY", raising=False)
    assert search_history_enabled() is True


@pytest.mark.asyncio
async def test_disabled_flag_writes_nothing(history_engine, monkeypatch):
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "false")
    payload = SimpleNamespace(dataset_id=uuid4(), completion="the answer")

    await log_search_history_mod.log_search_history("q?", "GRAPH_COMPLETION", uuid4(), [payload])

    assert await _row_counts(history_engine) == (0, 0)


@pytest.mark.asyncio
async def test_disabled_flag_skips_the_payload_loop(monkeypatch):
    """The gate sits ahead of log_query, so no Query object is even built."""
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "false")
    with (
        patch.object(log_search_history_mod, "log_query", AsyncMock()) as log_query,
        patch.object(log_search_history_mod, "log_result", AsyncMock()) as log_result,
    ):
        await log_search_history_mod.log_search_history("q?", "CHUNKS", uuid4(), [object()])

    log_query.assert_not_awaited()
    log_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_can_be_toggled_after_import(history_engine, monkeypatch):
    user_id = uuid4()
    payload = SimpleNamespace(dataset_id=None, completion="answer")

    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "false")
    await log_search_history_mod.log_search_history("q1?", "CHUNKS", user_id, [payload])
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    await log_search_history_mod.log_search_history("q2?", "CHUNKS", user_id, [payload])

    assert await _row_counts(history_engine) == (1, 1)


@pytest.mark.asyncio
async def test_delete_history_for_dataset_leaves_other_rows(history_engine, monkeypatch):
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    user_id = uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    for dataset_id in (dataset_a, dataset_b, None):
        query = await log_query_mod.log_query("q?", "CHUNKS", user_id, dataset_id)
        await log_result_mod.log_result(query.id, "a", user_id, dataset_id)

    removed = await delete_history_mod.delete_dataset_search_history(dataset_a)

    assert removed == {"queries_deleted": 1, "results_deleted": 1}
    remaining = await get_history_mod.get_history(user_id, limit=0)
    assert {row["dataset_id"] for row in remaining} == {dataset_b, None}


@pytest.mark.asyncio
async def test_delete_history_for_dataset_covers_every_user(history_engine, monkeypatch):
    """Owner A forgets a dataset user B searched through a read grant.

    The dataset is gone for B too, and B's answers can quote its chunks, so
    B's rows on that dataset go with it. B's other history is untouched.
    """
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    owner, reader = uuid4(), uuid4()
    shared, readers_own = uuid4(), uuid4()

    for user_id, dataset_id in ((owner, shared), (reader, shared), (reader, readers_own)):
        query = await log_query_mod.log_query("q?", "CHUNKS", user_id, dataset_id)
        await log_result_mod.log_result(query.id, "a", user_id, dataset_id)

    removed = await delete_history_mod.delete_dataset_search_history(shared)

    assert removed == {"queries_deleted": 2, "results_deleted": 2}
    assert await get_history_mod.get_history(owner, limit=0) == []
    remaining = await get_history_mod.get_history(reader, limit=0)
    assert {row["dataset_id"] for row in remaining} == {readers_own}


@pytest.mark.asyncio
async def test_delete_history_for_user_removes_unattributed_rows(history_engine, monkeypatch):
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    user_id = uuid4()
    for dataset_id in (uuid4(), None):
        query = await log_query_mod.log_query("q?", "CHUNKS", user_id, dataset_id)
        await log_result_mod.log_result(query.id, "a", user_id, dataset_id)

    removed = await delete_history_mod.delete_user_search_history(user_id)

    assert removed == {"queries_deleted": 2, "results_deleted": 2}
    assert await get_history_mod.get_history(user_id, limit=0) == []


@pytest.mark.asyncio
async def test_delete_history_ignores_the_logging_flag(history_engine, monkeypatch):
    """Rows written while logging was on are removable after it is turned off."""
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    user_id = uuid4()
    query = await log_query_mod.log_query("q?", "CHUNKS", user_id)
    await log_result_mod.log_result(query.id, "a", user_id)

    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "false")
    assert await delete_history_mod.delete_user_search_history(user_id) == {
        "queries_deleted": 1,
        "results_deleted": 1,
    }


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


async def _seed_history(user_id, dataset_id):
    query = await log_query_mod.log_query("q?", "CHUNKS", user_id, dataset_id)
    await log_result_mod.log_result(query.id, "a", user_id, dataset_id)


def _dataset_deletion_patches(dataset, graph_delete):
    """Stub the graph, lock and relational deletes around the real history purge."""
    return (
        patch.object(datasets_mod, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_mod, "dataset_lock", _no_op_context),
        patch.object(datasets_mod, "set_database_global_context_variables", _no_op_context),
        patch.object(datasets_mod, "delete_dataset_nodes_and_edges", graph_delete),
        patch.object(datasets_mod, "_invalidate_sessions_for_dataset_nonfatal", AsyncMock()),
        patch.object(datasets_mod, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()),
        patch("cognee.modules.data.methods.delete_dataset", AsyncMock()),
    )


@pytest.mark.asyncio
async def test_empty_dataset_purges_every_users_history_before_deleting(
    history_engine, monkeypatch
):
    """Owner A deletes a dataset reader B searched through a read grant.

    empty_dataset is behind forget(dataset), forget(everything) (via delete_all)
    and DELETE /api/v1/datasets/{id}, so all three purge here. B's rows on the
    dataset go with it, B's other history stays, and the purge is done before
    the graph is touched.
    """
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    owner, reader = uuid4(), uuid4()
    shared, readers_own = uuid4(), uuid4()
    for user_id, dataset_id in ((owner, shared), (reader, shared), (reader, readers_own)):
        await _seed_history(user_id, dataset_id)

    rows_left_when_graph_deleted = []

    async def graph_delete(*_args):
        rows_left_when_graph_deleted.append(await _row_counts(history_engine))

    dataset = SimpleNamespace(id=shared, owner_id=owner)
    with ExitStack() as stack:
        for context in _dataset_deletion_patches(dataset, graph_delete):
            stack.enter_context(context)
        await datasets_mod.datasets.empty_dataset(shared, user=SimpleNamespace(id=owner))

    assert rows_left_when_graph_deleted == [(1, 1)]
    assert await get_history_mod.get_history(owner, limit=0) == []
    remaining = await get_history_mod.get_history(reader, limit=0)
    assert {row["dataset_id"] for row in remaining} == {readers_own}


@pytest.mark.asyncio
async def test_empty_dataset_history_failure_deletes_nothing():
    """A failed purge raises before the graph or the dataset is touched, so the
    same delete can be retried."""
    owner = uuid4()
    dataset = SimpleNamespace(id=uuid4(), owner_id=owner)
    graph_delete = AsyncMock()

    with ExitStack() as stack:
        for context in _dataset_deletion_patches(dataset, graph_delete):
            stack.enter_context(context)
        stack.enter_context(
            patch(
                "cognee.modules.search.operations.delete_dataset_search_history",
                AsyncMock(side_effect=RuntimeError("history table missing")),
            )
        )
        with pytest.raises(RuntimeError, match="history table missing"):
            await datasets_mod.datasets.empty_dataset(dataset.id, user=SimpleNamespace(id=owner))

    graph_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_data_purges_history_when_it_deletes_the_emptied_dataset(
    history_engine, monkeypatch
):
    """delete_data(delete_dataset_if_empty=True) removing the last item deletes
    the dataset, so its history goes first, as in empty_dataset."""
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    owner = uuid4()
    dataset = SimpleNamespace(id=uuid4(), owner_id=owner)
    await _seed_history(owner, dataset.id)
    delete_dataset = AsyncMock()

    @asynccontextmanager
    async def operation(*_args, **_kwargs):
        yield SimpleNamespace(set_user=lambda _user: None)

    with (
        patch.object(datasets_mod, "record_operation", operation),
        patch.object(datasets_mod, "get_authorized_dataset", AsyncMock(return_value=dataset)),
        patch.object(datasets_mod, "dataset_lock", _no_op_context),
        patch.object(datasets_mod, "set_database_global_context_variables", _no_op_context),
        # The dataset is empty after the delete (one count query, no listing).
        patch.object(datasets_mod, "has_dataset_data", AsyncMock(return_value=False)),
        patch.object(
            datasets_mod,
            "delete_data_nodes_and_edges",
            AsyncMock(return_value=DeletedGraphElements()),
        ),
        patch.object(datasets_mod, "retire_forgotten_decisions", AsyncMock(return_value=set())),
        patch.object(datasets_mod, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()),
        patch("cognee.modules.data.methods.get_data", AsyncMock(return_value=None)),
        patch("cognee.modules.data.methods.delete_dataset", delete_dataset),
    ):
        await datasets_mod.datasets.delete_data(
            dataset.id, uuid4(), user=SimpleNamespace(id=owner), delete_dataset_if_empty=True
        )

    delete_dataset.assert_awaited_once_with(dataset)
    assert await _row_counts(history_engine) == (0, 0)


@pytest.mark.asyncio
async def test_forget_everything_purges_the_users_own_history_first():
    """The datasets' history goes inside empty_dataset; forget(everything) adds
    the user's own rows, rows saved without a dataset included, before delete_all."""
    user = SimpleNamespace(id=uuid4())
    calls = []
    fake_datasets = SimpleNamespace(
        list_datasets=AsyncMock(return_value=[object(), object()]),
        delete_all=AsyncMock(side_effect=lambda **_: calls.append("delete_all")),
    )
    delete_user_history = AsyncMock(side_effect=lambda _: calls.append("user_history") or COUNTS)

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch(
            "cognee.modules.search.operations.delete_user_search_history",
            delete_user_history,
        ),
        patch(
            "cognee.infrastructure.databases.cache.get_cache_config",
            return_value=SimpleNamespace(caching=False, usage_logging=False),
        ),
    ):
        result = await forget_mod._forget_everything(user)

    assert result == {
        "datasets_removed": 2,
        "search_history_removed": True,
        "history_cleaned": True,
        "status": "success",
    }
    delete_user_history.assert_awaited_once_with(user.id)
    assert calls == ["user_history", "delete_all"]


@pytest.mark.asyncio
async def test_forget_data_item_reports_history_kept():
    """History is attributed per dataset, so a document forget cannot purge it."""
    user = SimpleNamespace(id=uuid4())
    dataset_id, data_id = uuid4(), uuid4()
    # delete_data's receipt (DeleteDataReceiptDTO); forget passes it through and
    # adds the history flag.
    receipt = {"status": "success", "dataset_id": str(dataset_id), "data_id": str(data_id)}
    fake_datasets = SimpleNamespace(delete_data=AsyncMock(return_value=receipt))

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch.object(forget_mod, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch(
            "cognee.modules.search.operations.delete_dataset_search_history",
            AsyncMock(),
        ) as delete_dataset_history,
    ):
        result = await forget_mod._forget_data_item(data_id, dataset_id, user)

    assert result == {
        "data_id": str(data_id),
        "dataset_id": str(dataset_id),
        "search_history_removed": False,
        "status": "success",
    }
    delete_dataset_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_history_cleanup_failure_fails_forget_everything_before_the_data_goes():
    user = SimpleNamespace(id=uuid4())
    fake_datasets = SimpleNamespace(
        list_datasets=AsyncMock(return_value=[]), delete_all=AsyncMock()
    )

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch(
            "cognee.modules.search.operations.delete_user_search_history",
            AsyncMock(side_effect=RuntimeError("history table missing")),
        ),
        pytest.raises(RuntimeError, match="history table missing"),
    ):
        await forget_mod._forget_everything(user)

    fake_datasets.delete_all.assert_not_awaited()

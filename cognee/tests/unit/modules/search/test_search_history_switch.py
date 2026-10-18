"""``COGNEE_LOG_SEARCH_HISTORY`` and the forget paths that purge history.

The flag is read on every call, so it can be flipped after import; when it is
off nothing is written, not even the query row. ``delete_search_history``
ignores the flag: rows written while logging was on must still be removable.
"""

import importlib
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
    user_id, other_user = uuid4(), uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    for owner, dataset_id in ((user_id, dataset_a), (user_id, dataset_b), (user_id, None)):
        query = await log_query_mod.log_query("q?", "CHUNKS", owner, dataset_id)
        await log_result_mod.log_result(query.id, "a", owner, dataset_id)
    query = await log_query_mod.log_query("q?", "CHUNKS", other_user, dataset_a)
    await log_result_mod.log_result(query.id, "a", other_user, dataset_a)

    removed = await delete_history_mod.delete_search_history(user_id, dataset_a)

    assert removed == 2
    remaining = await get_history_mod.get_history(user_id, limit=0)
    assert {row["dataset_id"] for row in remaining} == {dataset_b, None}
    # Another user's rows on the same dataset are untouched.
    assert len(await get_history_mod.get_history(other_user, limit=0)) == 2


@pytest.mark.asyncio
async def test_delete_history_for_user_removes_unattributed_rows(history_engine, monkeypatch):
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    user_id = uuid4()
    for dataset_id in (uuid4(), None):
        query = await log_query_mod.log_query("q?", "CHUNKS", user_id, dataset_id)
        await log_result_mod.log_result(query.id, "a", user_id, dataset_id)

    removed = await delete_history_mod.delete_search_history(user_id)

    assert removed == 4
    assert await get_history_mod.get_history(user_id, limit=0) == []


@pytest.mark.asyncio
async def test_delete_history_ignores_the_logging_flag(history_engine, monkeypatch):
    """Rows written while logging was on are removable after it is turned off."""
    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "true")
    user_id = uuid4()
    query = await log_query_mod.log_query("q?", "CHUNKS", user_id)
    await log_result_mod.log_result(query.id, "a", user_id)

    monkeypatch.setenv("COGNEE_LOG_SEARCH_HISTORY", "false")
    assert await delete_history_mod.delete_search_history(user_id) == 2


@pytest.mark.asyncio
async def test_forget_everything_purges_user_history():
    user = SimpleNamespace(id=uuid4())
    fake_datasets = SimpleNamespace(
        list_datasets=AsyncMock(return_value=[object(), object()]),
        delete_all=AsyncMock(),
    )
    delete_history = AsyncMock(return_value=3)

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch(
            "cognee.modules.search.operations.delete_search_history",
            delete_history,
        ),
        patch(
            "cognee.infrastructure.databases.cache.get_cache_config",
            return_value=SimpleNamespace(caching=False, usage_logging=False),
        ),
    ):
        result = await forget_mod._forget_everything(user)

    assert result == {"datasets_removed": 2, "status": "success"}
    delete_history.assert_awaited_once_with(user.id, None)


@pytest.mark.asyncio
async def test_forget_dataset_purges_that_dataset_history():
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()
    fake_datasets = SimpleNamespace(empty_dataset=AsyncMock())
    delete_history = AsyncMock(return_value=1)

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch.object(forget_mod, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch(
            "cognee.modules.search.operations.delete_search_history",
            delete_history,
        ),
    ):
        result = await forget_mod._forget_dataset(dataset_id, user)

    assert result == {"dataset_id": str(dataset_id), "status": "success"}
    fake_datasets.empty_dataset.assert_awaited_once_with(dataset_id, user=user)
    delete_history.assert_awaited_once_with(user.id, dataset_id)


@pytest.mark.asyncio
async def test_history_cleanup_failure_does_not_fail_forget():
    """The data is already gone; an unreachable history table is logged, not raised."""
    user = SimpleNamespace(id=uuid4())
    dataset_id = uuid4()
    fake_datasets = SimpleNamespace(empty_dataset=AsyncMock())

    with (
        patch.object(datasets_mod, "datasets", fake_datasets),
        patch.object(forget_mod, "_resolve_dataset_id", AsyncMock(return_value=dataset_id)),
        patch(
            "cognee.modules.search.operations.delete_search_history",
            AsyncMock(side_effect=RuntimeError("history table missing")),
        ),
    ):
        result = await forget_mod._forget_dataset(dataset_id, user)

    assert result["status"] == "success"

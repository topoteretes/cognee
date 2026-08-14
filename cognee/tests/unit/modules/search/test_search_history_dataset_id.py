"""Round-trip coverage for ``dataset_id`` on recall/search history.

Exercises the real log -> read path against a temporary SQLite database:
``log_query``/``log_result`` write the column and ``get_history`` projects it
back on both sides of its UNION.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.search.models.Query import Query
from cognee.modules.search.models.Result import Result

log_query_mod = importlib.import_module("cognee.modules.search.operations.log_query")
log_result_mod = importlib.import_module("cognee.modules.search.operations.log_result")
get_history_mod = importlib.import_module("cognee.modules.search.operations.get_history")
search_mod = importlib.import_module("cognee.modules.search.methods.search")


@pytest_asyncio.fixture
async def history_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the two history tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="history_test.db",
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

    for module in (log_query_mod, log_result_mod, get_history_mod):
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)
        # Logging is gated on an env var read at import time; pin it on so the
        # test does not depend on the ambient environment.
        if hasattr(module, "_LOG_ENABLED"):
            monkeypatch.setattr(module, "_LOG_ENABLED", True)

    yield engine

    await engine.engine.dispose()


@pytest.mark.asyncio
async def test_dataset_id_round_trips_through_history(history_engine):
    user_id = uuid4()
    dataset_id = uuid4()

    query = await log_query_mod.log_query("who wrote it?", "GRAPH_COMPLETION", user_id, dataset_id)
    await log_result_mod.log_result(query.id, '["an answer"]', user_id, dataset_id)

    history = await get_history_mod.get_history(user_id, limit=0)

    assert len(history) == 2
    # Both the question and the answer carry the dataset attribution.
    assert {row["user"] for row in history} == {"user", "system"}
    assert all(row["dataset_id"] == dataset_id for row in history)


@pytest.mark.asyncio
async def test_unscoped_query_records_null_dataset_id(history_engine):
    user_id = uuid4()

    query = await log_query_mod.log_query("anything?", "GRAPH_COMPLETION", user_id)
    await log_result_mod.log_result(query.id, "[]", user_id)

    history = await get_history_mod.get_history(user_id, limit=0)

    assert len(history) == 2
    assert all(row["dataset_id"] is None for row in history)


@pytest.mark.asyncio
async def test_history_is_scoped_per_user(history_engine):
    """dataset_id must not leak history across users."""
    user_a, user_b = uuid4(), uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    await log_query_mod.log_query("a?", "GRAPH_COMPLETION", user_a, dataset_a)
    await log_query_mod.log_query("b?", "GRAPH_COMPLETION", user_b, dataset_b)

    history_a = await get_history_mod.get_history(user_a, limit=0)

    assert [row["dataset_id"] for row in history_a] == [dataset_a]


@pytest.mark.parametrize(
    "dataset_ids, expected_key",
    [
        (None, None),
        ([], None),
        (["single"], "single"),
        (["first", "second"], None),
    ],
)
def test_single_dataset_id_resolution(dataset_ids, expected_key):
    """Only an unambiguous single-dataset search is attributed."""
    ids = {"single": uuid4(), "first": uuid4(), "second": uuid4()}
    resolved = search_mod._single_dataset_id(
        None if dataset_ids is None else [ids[key] for key in dataset_ids]
    )

    assert resolved == (ids[expected_key] if expected_key else None)


def test_single_dataset_id_accepts_bare_uuid():
    """api/v1/search passes a bare UUID through when the caller gave one."""
    dataset_id = uuid4()

    assert search_mod._single_dataset_id(dataset_id) == dataset_id

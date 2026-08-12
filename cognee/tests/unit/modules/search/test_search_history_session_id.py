"""Round-trip coverage for ``session_id`` on recall/search history.

Same shape as ``test_search_history_dataset_id``: exercise the real log -> read
path against a temporary SQLite database. What matters beyond the round trip is
that ``log_search_history`` carries the session id onto **every** row it writes,
because a search that fans out over three datasets becomes three query rows and
all three belong to the same session.
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
log_history_mod = importlib.import_module("cognee.modules.search.operations.log_search_history")


class _Payload:
    """Minimal stand-in for a SearchResultPayload."""

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.completion = "an answer"


@pytest_asyncio.fixture
async def history_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the two history tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="history_session_test.db",
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
async def test_session_id_round_trips_through_history(history_engine):
    user_id = uuid4()

    query = await log_query_mod.log_query(
        "who wrote it?", "GRAPH_COMPLETION", user_id, None, "claude_ab12cd34"
    )
    await log_result_mod.log_result(query.id, '["an answer"]', user_id, None, "claude_ab12cd34")

    history = await get_history_mod.get_history(user_id, limit=0)

    assert len(history) == 2
    # Both the question and the answer carry the session attribution.
    assert {row["user"] for row in history} == {"user", "system"}
    assert all(row["session_id"] == "claude_ab12cd34" for row in history)


@pytest.mark.asyncio
async def test_caller_without_a_session_records_null(history_engine):
    """Raw API traffic sends no session id; that is NULL, not an error."""
    user_id = uuid4()

    query = await log_query_mod.log_query("anything?", "GRAPH_COMPLETION", user_id)
    await log_result_mod.log_result(query.id, "[]", user_id)

    history = await get_history_mod.get_history(user_id, limit=0)

    assert len(history) == 2
    assert all(row["session_id"] is None for row in history)


@pytest.mark.asyncio
async def test_every_fanned_out_row_carries_the_session(history_engine):
    """One search over three datasets is three rows, all in the same session."""
    user_id = uuid4()
    dataset_ids = [uuid4() for _ in range(3)]

    await log_history_mod.log_search_history(
        "what changed?",
        "GRAPH_COMPLETION",
        user_id,
        [_Payload(dataset_id) for dataset_id in dataset_ids],
        "codex_ef56ab78",
    )

    history = await get_history_mod.get_history(user_id, limit=0)

    # Three queries plus their three results.
    assert len(history) == 6
    assert all(row["session_id"] == "codex_ef56ab78" for row in history)
    assert {row["dataset_id"] for row in history} == set(dataset_ids)

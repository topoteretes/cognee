import asyncio
import errno
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.postgres import transient
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment


class ConnectionLost(Exception):
    sqlstate = "08003"


@pytest.mark.asyncio
async def test_transient_budget_preserves_terminal_error(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(transient.asyncio, "sleep", sleep)
    error = ConnectionLost("connection closed")
    operation = AsyncMock(side_effect=error)
    with pytest.raises(ConnectionLost) as raised:
        await transient.retry_postgres(operation)
    assert raised.value is error
    assert operation.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("invalid query"), asyncio.CancelledError()])
async def test_terminal_errors_and_cancellation_are_not_retried(error):
    operation = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await transient.retry_postgres(operation)
    assert operation.await_count == 1


def test_wrapped_driver_errors_and_cause_cycles():
    outer = RuntimeError("query failed")
    outer.__cause__ = OSError(errno.ECONNRESET, "reset")
    assert transient.is_transient_postgres_error(outer)
    outer.__cause__ = outer
    assert not transient.is_transient_postgres_error(outer)


@pytest.mark.asyncio
@pytest.mark.parametrize("neighborhood", [False, True])
async def test_unavailable_graph_is_not_reported_as_empty(monkeypatch, neighborhood):
    monkeypatch.setattr(transient.asyncio, "sleep", AsyncMock())
    project = AsyncMock(side_effect=ConnectionLost("connection closed"))
    method = "project_neighborhood_from_db" if neighborhood else "project_graph_from_db"
    monkeypatch.setattr(CogneeGraph, method, project)
    with pytest.raises(ConnectionLost):
        await get_memory_fragment(
            graph_engine=AsyncMock(),
            neighborhood_depth=1 if neighborhood else None,
            relevant_ids_to_filter=["seed"] if neighborhood else None,
        )
    assert project.await_count == 3

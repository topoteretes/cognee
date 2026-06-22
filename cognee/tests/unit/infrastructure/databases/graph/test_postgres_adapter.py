from contextlib import asynccontextmanager
import importlib.util
import sys
import types

import pytest

_created_asyncpg_stub = False
if importlib.util.find_spec("asyncpg") is None:
    asyncpg_stub = types.ModuleType("asyncpg")

    class DeadlockDetectedError(Exception):
        pass

    asyncpg_stub.DeadlockDetectedError = DeadlockDetectedError
    sys.modules["asyncpg"] = asyncpg_stub
    _created_asyncpg_stub = True

from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter  # noqa: E402

if _created_asyncpg_stub:
    del sys.modules["asyncpg"]


class _EmptyResult:
    def fetchall(self):
        return []


class _CapturingSession:
    def __init__(self):
        self.statement = None
        self.params = None

    async def execute(self, statement, params=None):
        self.statement = statement
        self.params = params
        return _EmptyResult()


@pytest.mark.asyncio
async def test_get_neighborhood_casts_seed_parameter_to_text_array():
    adapter = PostgresAdapter.__new__(PostgresAdapter)
    session = _CapturingSession()

    @asynccontextmanager
    async def session_context():
        yield session

    adapter._session = session_context

    nodes, edges = await adapter.get_neighborhood(["a"], depth=1)

    assert nodes == []
    assert edges == []
    assert "unnest(CAST(:seeds AS text[]))" in str(session.statement)
    assert session.params == {"seeds": ["a"], "depth": 1}

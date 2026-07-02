import pytest
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

try:
    from neo4j.exceptions import DatabaseUnavailable, Neo4jError
except ModuleNotFoundError:
    class Neo4jError(Exception):
        pass


    class DatabaseUnavailable(Exception):
        pass


    neo4j_module = types.ModuleType("neo4j")
    exceptions_module = types.ModuleType("neo4j.exceptions")
    exceptions_module.DatabaseUnavailable = DatabaseUnavailable
    exceptions_module.Neo4jError = Neo4jError
    neo4j_module.exceptions = exceptions_module
    sys.modules.setdefault("neo4j", neo4j_module)
    sys.modules["neo4j.exceptions"] = exceptions_module

from cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry import deadlock_retry


@pytest.mark.asyncio
async def test_deadlock_retry_errored():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(
        side_effect=[Neo4jError("DeadlockDetected"), Neo4jError("DeadlockDetected"), mock_return]
    )

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with pytest.raises(Neo4jError):
        await wrapped_function(self=None)


@pytest.mark.asyncio
async def test_deadlock_retry():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(side_effect=[Neo4jError("DeadlockDetected"), mock_return])

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeded on second time"


@pytest.mark.asyncio
async def test_deadlock_retry_exhaustive():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(
        side_effect=[Neo4jError("DeadlockDetected"), Neo4jError("DeadlockDetected"), mock_return]
    )

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeded on second time"


@pytest.mark.asyncio
async def test_deadlock_retry_allows_database_unavailable_retry(monkeypatch):
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(side_effect=[DatabaseUnavailable("temporary outage"), mock_return])

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    monkeypatch.setattr(
        "cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry.asyncio.sleep",
        AsyncMock(),
    )

    result = await wrapped_function(self=None)
    assert result, "Function should succeed after one DatabaseUnavailable retry"


if __name__ == "__main__":

    async def main():
        await test_deadlock_retry()
        await test_deadlock_retry_errored()
        await test_deadlock_retry_exhaustive()

    asyncio.run(main())

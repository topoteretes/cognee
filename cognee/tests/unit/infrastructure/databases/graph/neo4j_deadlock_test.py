import pytest
import asyncio
from unittest.mock import MagicMock
from neo4j.exceptions import Neo4jError
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


if __name__ == "__main__":

    async def main():
        await test_deadlock_retry()
        await test_deadlock_retry_errored()
        await test_deadlock_retry_exhaustive()

    asyncio.run(main())

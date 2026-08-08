import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from neo4j.exceptions import Neo4jError, DatabaseUnavailable
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
async def test_database_unavailable_retry_succeeds():
    """DatabaseUnavailable should retry and succeed within max_retries."""
    mock_function = AsyncMock(side_effect=[DatabaseUnavailable(), True])

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeeded after one DatabaseUnavailable retry"


@pytest.mark.asyncio
async def test_database_unavailable_exhausts_retries():
    """DatabaseUnavailable should re-raise once max_retries is exhausted."""
    mock_function = AsyncMock(
        side_effect=[DatabaseUnavailable(), DatabaseUnavailable()]
    )

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with pytest.raises(DatabaseUnavailable):
        await wrapped_function(self=None)


@pytest.mark.asyncio
async def test_database_unavailable_retries_match_neo4j_error():
    """DatabaseUnavailable and Neo4jError should both allow exactly max_retries attempts.

    This test directly validates the fix from issue #3757: the DatabaseUnavailable
    handler previously used >= max_retries (re-raising one attempt too early),
    while Neo4jError used > max_retries. Both now use > max_retries.
    """
    # With max_retries=2, we should be able to fail twice and succeed on the 3rd attempt.
    mock_function = AsyncMock(
        side_effect=[DatabaseUnavailable(), DatabaseUnavailable(), True]
    )

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeeded on the last allowed retry attempt"


if __name__ == "__main__":

    async def main():
        await test_deadlock_retry()
        await test_deadlock_retry_errored()
        await test_deadlock_retry_exhaustive()
        await test_database_unavailable_retry_succeeds()
        await test_database_unavailable_exhausts_retries()
        await test_database_unavailable_retries_match_neo4j_error()

    asyncio.run(main())

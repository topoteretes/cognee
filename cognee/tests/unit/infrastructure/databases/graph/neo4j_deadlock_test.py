import pytest
import asyncio
from unittest.mock import MagicMock, patch
from neo4j.exceptions import Neo4jError, DatabaseUnavailable
from cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry import deadlock_retry

# DatabaseUnavailable retries hit ``asyncio.sleep`` via ``calculate_backoff``.
# Patch the backoff to 0 so these tests stay fast and deterministic.
_no_backoff = patch(
    "cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry.calculate_backoff",
    return_value=0,
)


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
async def test_database_unavailable_is_retried():
    """DatabaseUnavailable must be retried, not raised on the first failure.

    Regression test: DatabaseUnavailable subclasses Neo4jError, so with the
    handlers ordered ``except Neo4jError`` first, the DatabaseUnavailable
    branch was unreachable and the error surfaced immediately (0 retries).
    """
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(side_effect=[DatabaseUnavailable("unavailable"), mock_return])

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with _no_backoff:
        result = await wrapped_function(self=None)

    assert result, "DatabaseUnavailable should be retried and then succeed"
    assert mock_function.call_count == 2


@pytest.mark.asyncio
async def test_database_unavailable_retry_count_matches_neo4jerror():
    """DatabaseUnavailable is retried the same number of times as Neo4jError.

    With max_retries=1 both allow exactly one retry (two calls total) before
    re-raising — previously DatabaseUnavailable used ``>=`` and raised one
    attempt earlier than the Neo4jError branch.
    """
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(
        side_effect=[
            DatabaseUnavailable("unavailable"),
            DatabaseUnavailable("unavailable"),
            mock_return,
        ]
    )

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with _no_backoff, pytest.raises(DatabaseUnavailable):
        await wrapped_function(self=None)

    assert mock_function.call_count == 2


if __name__ == "__main__":

    async def main():
        await test_deadlock_retry()
        await test_deadlock_retry_errored()
        await test_deadlock_retry_exhaustive()
        await test_database_unavailable_is_retried()
        await test_database_unavailable_retry_count_matches_neo4jerror()

    asyncio.run(main())

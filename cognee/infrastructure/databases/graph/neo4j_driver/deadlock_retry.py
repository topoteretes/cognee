import asyncio
from functools import wraps

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.utils.calculate_backoff import calculate_backoff


logger = get_logger("deadlock_retry")


def deadlock_retry(max_retries=10):
    """
    Decorator that automatically retries an asynchronous Neo4j operation on
    transient failures.

    Retries when the wrapped call raises ``DatabaseUnavailable`` or a Neo4j
    deadlock/transient error (``DeadlockDetected`` / ``Neo.TransientError``),
    using an exponential backoff with jitter between attempts. Any other
    ``Neo4jError`` is re-raised immediately.

    Args:
        max_retries: Maximum number of retry attempts after the initial call.

    Returns:
        The decorated async function.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            from neo4j.exceptions import Neo4jError, DatabaseUnavailable

            attempt = 0

            async def wait():
                backoff_time = calculate_backoff(attempt)
                logger.warning(
                    f"Neo4j transient error, retrying in {backoff_time:.2f}s "
                    f"(attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(backoff_time)

            while attempt <= max_retries:
                try:
                    attempt += 1
                    return await func(self, *args, **kwargs)
                except DatabaseUnavailable:
                    # DatabaseUnavailable subclasses Neo4jError, so this handler
                    # must precede the `except Neo4jError` below — otherwise the
                    # broader handler catches it first and this branch is dead
                    # code. Use the same `attempt > max_retries` bound as the
                    # Neo4jError branch so both retry the same number of times.
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    await wait()
                except Neo4jError as error:
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    error_str = str(error)
                    if "DeadlockDetected" in error_str or "Neo.TransientError" in error_str:
                        await wait()
                    else:
                        raise  # Re-raise the original error

        return wrapper

    return decorator

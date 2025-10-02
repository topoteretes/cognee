import asyncio
from functools import wraps

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.utils.calculate_backoff import calculate_backoff


logger = get_logger("deadlock_retry")


def deadlock_retry(max_retries=10):
    """
    Decorator that automatically retries an asynchronous function when rate limit errors occur.

    This decorator implements an exponential backoff strategy with jitter
    to handle rate limit errors efficiently.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_backoff: Initial backoff time in seconds.
        backoff_factor: Multiplier for exponential backoff.
        jitter: Jitter factor to avoid the thundering herd problem.

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
                    f"Neo4j rate limit hit, retrying in {backoff_time:.2f}s "
                    f"Attempt {attempt}/{max_retries}"
                )
                await asyncio.sleep(backoff_time)

            while attempt <= max_retries:
                try:
                    attempt += 1
                    return await func(self, *args, **kwargs)
                except Neo4jError as error:
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    error_str = str(error)
                    if "DeadlockDetected" in error_str or "Neo.TransientError" in error_str:
                        await wait()
                    else:
                        raise  # Re-raise the original error
                except DatabaseUnavailable:
                    if attempt >= max_retries:
                        raise  # Re-raise the original error

                    await wait()

        return wrapper

    return decorator

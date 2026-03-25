import asyncio
from typing import Any, Coroutine, List


async def gather_with_concurrency(max_concurrent: int, *coros: Coroutine) -> List[Any]:
    """Like asyncio.gather but limits concurrency via semaphore.

    Args:
        max_concurrent: Maximum number of coroutines running at the same time.
        *coros: Coroutines to execute.

    Returns:
        List of results in the same order as the input coroutines.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro: Coroutine) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[limited(c) for c in coros])

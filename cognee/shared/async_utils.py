import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

AsyncCallFactory = Callable[[], Awaitable[T]]


async def gather_with_concurrency_limit(
    call_factories: Iterable[AsyncCallFactory[T]],
    limit: int,
) -> list[T]:
    """Run async call factories with at most ``limit`` calls in flight.

    Results preserve the input order, matching ``asyncio.gather``.
    """
    if limit < 1:
        raise ValueError("limit must be positive")

    semaphore = asyncio.Semaphore(limit)

    async def run(call_factory: AsyncCallFactory[T]) -> T:
        async with semaphore:
            return await call_factory()

    return list(await asyncio.gather(*(run(call_factory) for call_factory in call_factories)))

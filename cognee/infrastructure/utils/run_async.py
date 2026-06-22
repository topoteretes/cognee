import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import Executor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


async def run_async(
    func: Callable[..., T],
    *args: Any,
    loop: asyncio.AbstractEventLoop | None = None,
    executor: Executor | None = None,
    **kwargs: Any,
) -> T:
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

    if "loop" in inspect.signature(func).parameters:
        pfunc = partial(func, *args, loop=loop, **kwargs)
    else:
        pfunc = partial(func, *args, **kwargs)

    return await loop.run_in_executor(executor, pfunc)

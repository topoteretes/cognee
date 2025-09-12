import asyncio
from functools import partial
import inspect


async def run_async(func, *args, loop=None, executor=None, **kwargs):
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

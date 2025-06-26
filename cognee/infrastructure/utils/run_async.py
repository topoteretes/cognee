import asyncio
from functools import partial


async def run_async(func, *args, loop=None, executor=None, **kwargs):
    if loop is None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = asyncio.get_event_loop()

    pfunc = partial(func, *args, **kwargs)
    return await running_loop.run_in_executor(executor, pfunc)

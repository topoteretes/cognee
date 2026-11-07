import asyncio
import contextvars
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

T = TypeVar("T")


def _new_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="cognee-cpu")


# One thread for the whole process, not the loop's default executor. CPU-bound Python
# holds the GIL, so more threads would not finish the work sooner: several documents
# cutting chunks at once on their own threads only contend for the GIL, and they also
# fill the default executor that DNS lookups and file I/O wait on.
_executor = _new_executor()


def _replace_executor_after_fork() -> None:
    # A forked child inherits the executor but not its thread.
    global _executor
    _executor = _new_executor()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_replace_executor_after_fork)


async def run_cpu_bound(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run CPU-bound ``func(*args, **kwargs)`` on the shared worker thread and await it.

    Calls from every pipeline take turns on that one thread, so the event loop stays
    free while they run. Like ``asyncio.to_thread``, each call runs in a copy of the
    caller's context, so ContextVars such as the per-call embedding config still apply.
    """
    loop = asyncio.get_running_loop()
    call = partial(contextvars.copy_context().run, func, *args, **kwargs)
    return await loop.run_in_executor(_executor, call)

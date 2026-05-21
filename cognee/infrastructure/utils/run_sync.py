import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(
    coro: Coroutine[Any, Any, T],
    running_loop: asyncio.AbstractEventLoop | None = None,
    timeout: float | None = None,
) -> T | None:
    result = None
    exception = None

    def runner() -> None:
        nonlocal result, exception, running_loop

        try:
            try:
                if not running_loop:
                    running_loop = asyncio.get_running_loop()

                result = asyncio.run_coroutine_threadsafe(coro, running_loop).result(timeout)
            except RuntimeError:
                result = asyncio.run(coro)

        except Exception as e:
            exception = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise asyncio.TimeoutError("Coroutine execution timed out.")
    if exception:
        raise exception

    return result

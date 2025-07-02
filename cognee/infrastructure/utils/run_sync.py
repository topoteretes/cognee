import asyncio
import threading


def run_sync(coro, timeout=None):
    result = None
    exception = None

    def runner():
        nonlocal result, exception
        try:
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

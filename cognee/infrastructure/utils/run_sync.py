import asyncio
import threading


def run_sync(coro, timeout=None):
    result = None
    exception = None

    def runner():
        nonlocal result, exception

        try:
            try:
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

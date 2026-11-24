"""run_cpu_bound: CPU-bound calls take turns on one worker thread, in the caller's context."""

import asyncio
import threading
from contextvars import ContextVar

import pytest

from cognee.infrastructure.utils.run_cpu_bound import run_cpu_bound

_request = ContextVar("request", default=None)


@pytest.mark.asyncio
async def test_calls_take_turns_on_one_worker_thread():
    """A second call waits for the first: more threads would only contend for the GIL."""
    first_started = threading.Event()
    release_first = threading.Event()
    threads, finished = [], []

    def first():
        threads.append(threading.current_thread())
        first_started.set()
        assert release_first.wait(timeout=5), "the test never released the first call"
        finished.append("first")

    def second():
        threads.append(threading.current_thread())
        finished.append("second")

    calls = asyncio.gather(run_cpu_bound(first), run_cpu_bound(second))
    assert await asyncio.to_thread(first_started.wait, 5)
    await asyncio.sleep(0.05)  # room for a second worker, if there were one, to run `second`
    assert finished == []
    release_first.set()
    await calls

    assert finished == ["first", "second"]
    assert threads[0] is threads[1]
    assert threads[0] is not threading.current_thread()


@pytest.mark.asyncio
async def test_runs_in_a_copy_of_the_callers_context():
    def read_then_overwrite():
        seen = _request.get()
        _request.set("set on the worker")
        return seen

    token = _request.set("set by the caller")
    try:
        assert await run_cpu_bound(read_then_overwrite) == "set by the caller"
        assert _request.get() == "set by the caller"
    finally:
        _request.reset(token)


@pytest.mark.asyncio
async def test_passes_arguments_and_returns_the_result():
    assert await run_cpu_bound(int, "ff", base=16) == 255


@pytest.mark.asyncio
async def test_errors_reach_the_caller():
    def fail():
        raise ValueError("word longer than the chunk size")

    with pytest.raises(ValueError, match="word longer than the chunk size"):
        await run_cpu_bound(fail)

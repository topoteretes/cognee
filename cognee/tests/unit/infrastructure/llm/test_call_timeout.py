import asyncio

import pytest

from cognee.infrastructure.llm.call_timeout import run_with_timeout
from cognee.infrastructure.llm.exceptions import LLMCallTimeoutError


@pytest.mark.asyncio
async def test_run_with_timeout_returns_result():
    async def complete():
        return "done"

    assert (
        await run_with_timeout(complete(), timeout_seconds=1, operation="test completion") == "done"
    )


@pytest.mark.asyncio
async def test_run_with_timeout_preserves_provider_exception():
    provider_error = RuntimeError("provider failed")

    async def fail():
        raise provider_error

    with pytest.raises(RuntimeError) as raised:
        await run_with_timeout(fail(), timeout_seconds=1, operation="test completion")

    assert raised.value is provider_error


@pytest.mark.asyncio
async def test_run_with_timeout_does_not_wait_for_cancellation_cleanup():
    cleanup_finished = asyncio.Event()

    async def delay_cancellation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
            cleanup_finished.set()
            raise

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMCallTimeoutError, match="0.02 second") as raised:
        await run_with_timeout(
            delay_cancellation(), timeout_seconds=0.02, operation="test completion"
        )

    assert loop.time() - started < 0.15
    assert raised.value.operation == "test completion"
    assert raised.value.timeout_seconds == 0.02
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1)

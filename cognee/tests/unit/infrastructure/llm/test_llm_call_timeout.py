import asyncio

import pytest

from cognee.infrastructure.llm.call_timeout import (
    LLMCallTimeoutError,
    build_llm_call_label,
    run_with_llm_call_limits,
)
from cognee.infrastructure.llm.config import LLMConfig


@pytest.mark.asyncio
async def test_run_with_llm_call_limits_disabled_by_default():
    async def fast_call():
        await asyncio.sleep(0)
        return "ok"

    result = await run_with_llm_call_limits(
        fast_call(),
        label="provider=custom model=test endpoint=http://localhost:8000/v1",
        timeout_seconds=0,
        slow_warning_seconds=0,
    )
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_with_llm_call_limits_raises_when_opt_in_timeout_exceeded():
    async def slow_call():
        await asyncio.sleep(0.2)
        return "late"

    with pytest.raises(LLMCallTimeoutError, match="LLM_CALL_TIMEOUT_SECONDS=0.1"):
        await run_with_llm_call_limits(
            slow_call(),
            label=build_llm_call_label(
                provider="custom",
                model="openai/test-model",
                endpoint="http://localhost:8000/v1",
            ),
            timeout_seconds=0.1,
            slow_warning_seconds=0,
        )


@pytest.mark.asyncio
async def test_run_with_llm_call_limits_continues_after_slow_warning_threshold():
    async def medium_call():
        await asyncio.sleep(0.15)
        return "done"

    result = await run_with_llm_call_limits(
        medium_call(),
        label="provider=custom model=test endpoint=http://localhost:8000/v1",
        timeout_seconds=0,
        slow_warning_seconds=0.05,
    )

    assert result == "done"


def test_llm_config_defaults_keep_timeout_disabled():
    config = LLMConfig(
        llm_provider="custom",
        llm_model="openai/test-model",
        llm_api_key="test-key",
        llm_endpoint="http://localhost:8000/v1",
    )
    assert config.llm_call_timeout_seconds == 0
    assert config.llm_slow_call_warning_seconds == 60

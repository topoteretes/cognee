"""Wiring test: BedrockAdapter dispatches through the pacing seam.

The overload policy only sees errors raised inside
``llm_rate_limiter_context_manager()``; this locks the Bedrock adapter to that
seam so a recognized overload error paces the attempts that follow.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from pydantic import BaseModel

import cognee.infrastructure.llm.config as llm_config_module
import cognee.infrastructure.llm.overload_policy as overload_policy_module
from cognee.infrastructure.llm.overload_policy import OverloadPolicy
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
    BedrockAdapter,
)
from cognee.shared import rate_limiting
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager


class PacerSpy:
    def __init__(self):
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def seam(monkeypatch):
    policy = OverloadPolicy()
    monkeypatch.setattr(overload_policy_module, "llm_overload_policy", policy)
    spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "_llm_rate_limiter", spy)
    monkeypatch.setattr(
        llm_config_module,
        "get_llm_config",
        lambda: SimpleNamespace(
            llm_rate_limit_enabled=False,
            auto_rate_limit=True,
            llm_rate_limit_requests=60,
            llm_rate_limit_interval=60,
        ),
    )
    return SimpleNamespace(policy=policy, spy=spy)


class _Answer(BaseModel):
    text: str


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def _unwrapped(fn):
    """The undecorated method body — skips tenacity's real backoff waits."""
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@pytest.mark.asyncio
async def test_bedrock_overload_error_reaches_policy_and_paces_next_dispatch(seam, monkeypatch):
    adapter = BedrockAdapter(model="anthropic.claude-3-haiku", api_key="test-key")
    monkeypatch.setattr(
        adapter.aclient.chat.completions,
        "create",
        AsyncMock(side_effect=_rate_limit_error()),
    )

    body = _unwrapped(BedrockAdapter.acreate_structured_output)
    with pytest.raises(openai.RateLimitError):
        await body(adapter, "text", "prompt", _Answer)

    assert seam.policy.is_paced() is True

    async with llm_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 1  # the episode paces subsequent dispatches

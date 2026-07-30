"""Seam tests: the dispatch context manager wires config, pacing, and the overload policy."""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

import cognee.infrastructure.llm.config as llm_config_module
import cognee.infrastructure.llm.overload_policy as overload_policy_module
from cognee.infrastructure.llm.overload_policy import OverloadPolicy
from cognee.shared import rate_limiting
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager


class PacerSpy:
    """Stands in for the AsyncLimiter; records whether pacing was applied."""

    def __init__(self):
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        return False


def _config(**overrides):
    defaults = dict(
        llm_rate_limit_enabled=False,
        auto_rate_limit=True,
        llm_rate_limit_requests=60,
        llm_rate_limit_interval=60,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def seam(monkeypatch):
    """Fresh policy + pacer spy + config stub for every test."""
    policy = OverloadPolicy()
    monkeypatch.setattr(overload_policy_module, "llm_overload_policy", policy)
    spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "_llm_rate_limiter", spy)
    monkeypatch.setattr(llm_config_module, "get_llm_config", lambda: _config())
    return SimpleNamespace(policy=policy, spy=spy)


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.mark.asyncio
async def test_full_speed_by_default(seam):
    async with llm_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 0
    assert seam.policy.is_paced() is False


@pytest.mark.asyncio
async def test_overload_error_reaches_the_policy_and_paces_next_dispatch(seam):
    with pytest.raises(openai.RateLimitError):
        async with llm_rate_limiter_context_manager():
            raise _rate_limit_error()
    assert seam.policy.is_paced() is True

    async with llm_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 1  # paced while the episode lasts


@pytest.mark.asyncio
async def test_auto_rate_limit_off_bypasses_the_policy(seam, monkeypatch):
    monkeypatch.setattr(llm_config_module, "get_llm_config", lambda: _config(auto_rate_limit=False))
    with pytest.raises(openai.RateLimitError):
        async with llm_rate_limiter_context_manager():
            raise _rate_limit_error()
    assert seam.policy.is_paced() is False
    assert seam.spy.entered == 0


@pytest.mark.asyncio
async def test_explicit_rate_limit_setting_paces_from_the_start(seam, monkeypatch):
    monkeypatch.setattr(
        llm_config_module, "get_llm_config", lambda: _config(llm_rate_limit_enabled=True)
    )
    async with llm_rate_limiter_context_manager():
        pass
    assert seam.spy.entered == 1

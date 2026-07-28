"""Tests for AUTO_RATE_LIMIT: unbounded dispatch until a provider rate limit."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import openai
import pytest

from cognee.shared import rate_limiting
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager


class PacerSpy:
    """Stands in for the AsyncLimiter; records whether pacing was applied."""

    def __init__(self):
        self.entered = 0

    @asynccontextmanager
    async def _cm(self):
        self.entered += 1
        yield

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
        llm_provider="openai",
        llm_model="openai/gpt-5-mini",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Reset activation; default config: cloud provider, rate limiting off, auto on."""
    monkeypatch.setattr(rate_limiting, "_auto_rate_limit_activated", False)
    monkeypatch.setattr(rate_limiting, "_local_pacing_active", None)
    monkeypatch.setattr(rate_limiting, "llm_config", _config())
    spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "llm_rate_limiter", spy)
    return spy


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.mark.asyncio
async def test_unbounded_and_unpaced_by_default(fresh_state):
    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 0
    assert rate_limiting._auto_rate_limit_activated is False


@pytest.mark.asyncio
async def test_rate_limit_error_activates_pacing_with_warning(fresh_state, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        rate_limiting,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )

    # Wrapped the way instructor surfaces provider errors: via the cause chain.
    wrapper = RuntimeError("instructor wrapper")
    wrapper.__cause__ = _rate_limit_error()
    with pytest.raises(RuntimeError):
        async with llm_rate_limiter_context_manager():
            raise wrapper

    assert rate_limiting._auto_rate_limit_activated is True
    assert len(warnings) == 1 and "rate limit" in warnings[0]

    # Subsequent dispatches are paced; a repeat rate limit does not re-warn.
    with pytest.raises(openai.RateLimitError):
        async with llm_rate_limiter_context_manager():
            raise _rate_limit_error()
    assert fresh_state.entered == 1
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_auto_rate_limit_disabled_never_activates(fresh_state, monkeypatch):
    monkeypatch.setattr(rate_limiting, "llm_config", _config(auto_rate_limit=False))
    with pytest.raises(openai.RateLimitError):
        async with llm_rate_limiter_context_manager():
            raise _rate_limit_error()
    assert rate_limiting._auto_rate_limit_activated is False
    assert fresh_state.entered == 0


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("ollama", "phi4:latest"),
        ("llama_cpp", "some-model"),
        ("custom", "lm_studio/qwen2.5-7b"),
        ("custom", "hosted_vllm/meta-llama/Llama-3-70B"),
    ],
)
@pytest.mark.asyncio
async def test_local_provider_paced_from_first_dispatch(fresh_state, monkeypatch, provider, model):
    warnings = []
    monkeypatch.setattr(
        rate_limiting,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    monkeypatch.setattr(
        rate_limiting, "llm_config", _config(llm_provider=provider, llm_model=model)
    )

    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 1  # paced immediately, no error needed
    assert len(warnings) == 1 and "Local LLM provider detected" in warnings[0]

    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 2
    assert len(warnings) == 1  # warned exactly once


@pytest.mark.asyncio
async def test_local_pacing_disabled_by_auto_rate_limit_flag(fresh_state, monkeypatch):
    monkeypatch.setattr(
        rate_limiting,
        "llm_config",
        _config(llm_provider="ollama", llm_model="phi4:latest", auto_rate_limit=False),
    )
    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 0


@pytest.mark.asyncio
async def test_non_rate_limit_errors_do_not_activate(fresh_state):
    with pytest.raises(ValueError):
        async with llm_rate_limiter_context_manager():
            raise ValueError("schema mismatch")
    assert rate_limiting._auto_rate_limit_activated is False


@pytest.mark.asyncio
async def test_explicit_rate_limit_setting_paces_from_the_start(fresh_state, monkeypatch):
    monkeypatch.setattr(
        rate_limiting,
        "llm_config",
        SimpleNamespace(llm_rate_limit_enabled=True, auto_rate_limit=True),
    )
    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 1

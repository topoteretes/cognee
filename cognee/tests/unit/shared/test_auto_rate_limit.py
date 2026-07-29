"""Tests for AUTO_RATE_LIMIT: full speed until overload, then warn and enable the RPM limiter."""

import asyncio
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
        model_fields_set=set(),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Reset activation; default config: rate limiting off, auto fallback on."""
    monkeypatch.setattr(rate_limiting, "_auto_rate_limit_activated", False)
    monkeypatch.setattr(rate_limiting, "llm_config", _config())
    spy = PacerSpy()
    monkeypatch.setattr(rate_limiting, "llm_rate_limiter", spy)
    return spy


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://api.test"))


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "http://api.test"))
    return openai.APIStatusError("overloaded", response=response, body=None)


@pytest.mark.asyncio
async def test_full_speed_by_default(fresh_state):
    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 0
    assert rate_limiting._auto_rate_limit_activated is False


@pytest.mark.parametrize(
    ("overload_error", "expected_name"),
    [
        (_rate_limit_error(), "RateLimitError"),
        (_timeout_error(), "APITimeoutError"),
        (asyncio.TimeoutError(), "TimeoutError"),
        (_status_error(503), "HTTP 503"),
        (_status_error(529), "HTTP 529"),
    ],
)
@pytest.mark.asyncio
async def test_overload_enables_rate_limiter_with_warning(
    fresh_state, monkeypatch, overload_error, expected_name
):
    warnings = []
    monkeypatch.setattr(
        rate_limiting,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )

    # Wrapped the way instructor surfaces provider errors: via the cause chain.
    wrapper = RuntimeError("instructor wrapper")
    wrapper.__cause__ = overload_error
    with pytest.raises(RuntimeError):
        async with llm_rate_limiter_context_manager():
            raise wrapper

    assert rate_limiting._auto_rate_limit_activated is True
    assert len(warnings) == 1
    assert "not being processed fast enough" in warnings[0]
    assert expected_name in warnings[0]

    # Subsequent dispatches are paced; repeat overload does not re-warn.
    with pytest.raises(openai.RateLimitError):
        async with llm_rate_limiter_context_manager():
            raise _rate_limit_error()
    assert fresh_state.entered == 1
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_slow_completion_enables_rate_limiter_before_any_failure(fresh_state, monkeypatch):
    monkeypatch.setattr(rate_limiting, "SLOW_COMPLETION_SECONDS", 0.05)
    warnings = []
    monkeypatch.setattr(
        rate_limiting,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    async with llm_rate_limiter_context_manager():
        await asyncio.sleep(0.06)  # a successful but dangerously slow call
    assert rate_limiting._auto_rate_limit_activated is True
    assert len(warnings) == 1 and "completion took" in warnings[0]


@pytest.mark.asyncio
async def test_auto_rate_limit_disabled_never_activates(fresh_state, monkeypatch):
    monkeypatch.setattr(rate_limiting, "llm_config", _config(auto_rate_limit=False))
    for error in (_rate_limit_error(), _timeout_error(), _status_error(503)):
        with pytest.raises(type(error)):
            async with llm_rate_limiter_context_manager():
                raise error
    assert rate_limiting._auto_rate_limit_activated is False
    assert fresh_state.entered == 0


@pytest.mark.asyncio
async def test_ordinary_errors_do_not_activate(fresh_state):
    with pytest.raises(ValueError):
        async with llm_rate_limiter_context_manager():
            raise ValueError("schema mismatch")
    with pytest.raises(openai.APIStatusError):
        async with llm_rate_limiter_context_manager():
            raise _status_error(500)  # a plain server error is not overload evidence
    assert rate_limiting._auto_rate_limit_activated is False


@pytest.mark.asyncio
async def test_explicit_rate_limit_setting_paces_from_the_start(fresh_state, monkeypatch):
    monkeypatch.setattr(rate_limiting, "llm_config", _config(llm_rate_limit_enabled=True))
    async with llm_rate_limiter_context_manager():
        pass
    assert fresh_state.entered == 1

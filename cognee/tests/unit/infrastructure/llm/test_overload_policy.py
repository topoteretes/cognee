"""Tests for the LLM overload policy: evidence taxonomy and cooldown episodes."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

import cognee.infrastructure.llm.overload_policy as overload_policy_module
from cognee.infrastructure.llm.overload_policy import OverloadPolicy, overload_evidence


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://api.test"))


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "http://api.test"))
    return openai.APIStatusError("overloaded", response=response, body=None)


# --------------------------------------------------------------------------- #
# Evidence taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_rate_limit_error(), "RateLimitError"),
        (_timeout_error(), "APITimeoutError"),
        (asyncio.TimeoutError(), "TimeoutError"),
        (_status_error(503), "HTTP 503"),
        (_status_error(529), "HTTP 529"),
    ],
)
def test_overload_evidence_names_the_cause(error, expected):
    assert overload_evidence(error) == expected


def test_overload_evidence_sees_through_wrappers():
    wrapper = RuntimeError("instructor wrapper")  # errors arrive wrapped in practice
    wrapper.__cause__ = RuntimeError("retry wrapper")
    wrapper.__cause__.__cause__ = _rate_limit_error()
    assert overload_evidence(wrapper) == "RateLimitError"


@pytest.mark.parametrize(
    "error",
    [ValueError("schema mismatch"), _status_error(500)],
)
def test_ordinary_errors_are_not_evidence(error):
    assert overload_evidence(error) is None


# --------------------------------------------------------------------------- #
# Policy episodes
# --------------------------------------------------------------------------- #


@pytest.fixture
def warnings(monkeypatch):
    captured = []
    monkeypatch.setattr(
        overload_policy_module,
        "logger",
        SimpleNamespace(warning=lambda message, *args: captured.append(message % args)),
    )
    return captured


def test_unpaced_until_evidence(warnings):
    policy = OverloadPolicy()
    assert policy.is_paced() is False

    policy.on_error(ValueError("not overload"))
    assert policy.is_paced() is False
    assert warnings == []


def test_overload_error_starts_a_warned_episode(warnings):
    policy = OverloadPolicy()
    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True
    assert len(warnings) == 1 and "Potential RPM issues detected" in warnings[0]


def test_evidence_extends_episode_without_rewarning(warnings):
    policy = OverloadPolicy(cooldown_seconds=5.0)
    policy.on_error(_rate_limit_error())
    first_deadline = policy._paced_until

    time.sleep(0.01)
    policy.on_error(_timeout_error())
    assert policy._paced_until > first_deadline
    assert len(warnings) == 1  # same episode: one warning


def test_episode_lapses_and_a_fresh_one_rewarns(warnings):
    policy = OverloadPolicy(cooldown_seconds=0.05)
    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True

    time.sleep(0.06)
    assert policy.is_paced() is False  # lapsed back to the configured state

    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True
    assert len(warnings) == 2  # fresh episode warns again


# --------------------------------------------------------------------------- #
# Provider-neutral evidence: native SDK errors outside the openai hierarchy
# --------------------------------------------------------------------------- #


class _DuckStatusError(Exception):
    """Any exception exposing status_code, like every provider SDK's errors."""

    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, "HTTP 429"), (503, "HTTP 503"), (529, "HTTP 529"), (500, None), (400, None)],
)
def test_status_codes_are_recognized_on_any_exception(status, expected):
    assert overload_evidence(_DuckStatusError(status)) == expected


def test_httpx_timeout_is_evidence():
    assert overload_evidence(httpx.ReadTimeout("read timed out")) == "ReadTimeout"


def _native_anthropic_errors():
    """Native Anthropic errors, built the way the SDK raises them."""
    anthropic = pytest.importorskip("anthropic")
    request = httpx.Request("POST", "http://api.test")

    def status_error(status: int, cls):
        return cls("overloaded", response=httpx.Response(status, request=request), body=None)

    # The SDK raises APITimeoutError `from` the underlying httpx timeout;
    # evidence detection walks the cause chain to the transport error.
    timeout = anthropic.APITimeoutError(request=request)
    timeout.__cause__ = httpx.ConnectTimeout("connect timed out")

    return [
        (status_error(429, anthropic.RateLimitError), "HTTP 429"),
        (status_error(503, anthropic.APIStatusError), "HTTP 503"),
        (status_error(529, anthropic.APIStatusError), "HTTP 529"),
        (timeout, "ConnectTimeout"),
        (status_error(400, anthropic.APIStatusError), None),
    ]


def test_native_anthropic_errors_are_recognized():
    for error, expected in _native_anthropic_errors():
        assert overload_evidence(error) == expected, type(error).__name__


def test_instructor_wrapped_native_anthropic_error_is_recognized():
    anthropic = pytest.importorskip("anthropic")
    request = httpx.Request("POST", "http://api.test")
    native = anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )
    wrapper = RuntimeError("instructor wrapper")  # instructor re-raises with the cause chained
    wrapper.__cause__ = native
    assert overload_evidence(wrapper) == "HTTP 429"

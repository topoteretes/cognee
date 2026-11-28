"""Tests for the shared overload policy: evidence taxonomy and throttling periods."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

import cognee.shared.overload_policy as overload_policy_module
from cognee.shared.overload_policy import OverloadPolicy, PacingBudget, overload_evidence


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://api.test"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def _timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://api.test"))


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "http://api.test"))
    return openai.APIStatusError("overloaded", response=response, body=None)


def _budget() -> PacingBudget:
    return PacingBudget(60, 60, "TEST_RATE_LIMIT_REQUESTS", "TEST_RATE_LIMIT_INTERVAL")


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
# Throttling periods
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
    policy = OverloadPolicy(_budget)
    assert policy.is_paced() is False

    policy.on_error(ValueError("not overload"))
    assert policy.is_paced() is False
    assert warnings == []


def test_overload_error_starts_warned_throttling(warnings):
    policy = OverloadPolicy(_budget)
    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True
    assert len(warnings) == 1 and "Potential RPM issues detected" in warnings[0]
    assert "TEST_RATE_LIMIT_REQUESTS / TEST_RATE_LIMIT_INTERVAL" in warnings[0]


def test_evidence_extends_throttling_without_rewarning(warnings):
    policy = OverloadPolicy(_budget, cooldown_seconds=5.0)
    policy.on_error(_rate_limit_error())
    first_deadline = policy._paced_until

    time.sleep(0.01)
    policy.on_error(_timeout_error())
    assert policy._paced_until > first_deadline
    assert len(warnings) == 1  # same throttling period: one warning


def test_throttling_lapses_and_new_throttling_rewarns(warnings):
    policy = OverloadPolicy(_budget, cooldown_seconds=0.05)
    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True

    time.sleep(0.06)
    assert policy.is_paced() is False  # lapsed back to the configured state

    policy.on_error(_rate_limit_error())
    assert policy.is_paced() is True
    assert len(warnings) == 2  # new throttling period warns again


def test_embedding_budget_warning_names_embedding_settings(warnings):
    from cognee.infrastructure.databases.vector.embeddings.config import (
        embedding_pacing_budget,
    )

    policy = OverloadPolicy(embedding_pacing_budget)
    policy.on_error(_rate_limit_error())
    assert len(warnings) == 1
    assert "EMBEDDING_RATE_LIMIT_REQUESTS / EMBEDDING_RATE_LIMIT_INTERVAL" in warnings[0]


@pytest.mark.parametrize("paced_side", ["llm", "embedding"])
def test_evidence_on_one_side_leaves_the_other_unpaced(paced_side, warnings, monkeypatch):
    from cognee.infrastructure.databases.vector.embeddings.config import (
        embedding_overload_policy,
    )
    from cognee.infrastructure.llm.config import llm_overload_policy

    policies = {"llm": llm_overload_policy, "embedding": embedding_overload_policy}
    for policy in policies.values():
        monkeypatch.setattr(policy, "_paced_until", 0.0)

    policies[paced_side].on_error(_rate_limit_error())

    other_side = "embedding" if paced_side == "llm" else "llm"
    assert policies[paced_side].is_paced() is True
    assert policies[other_side].is_paced() is False


def test_embedding_throttling_lasts_60_seconds_and_llm_throttling_900():
    from cognee.infrastructure.databases.vector.embeddings.config import (
        embedding_overload_policy,
    )
    from cognee.infrastructure.llm.config import llm_overload_policy

    assert embedding_overload_policy.cooldown_seconds == 60.0
    assert llm_overload_policy.cooldown_seconds == 900.0


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

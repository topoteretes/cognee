"""Embedding retries wait for the provider's Retry-After, within the retry window."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import openai

from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.databases.vector.embeddings.retry_config import (
    stop_when_retry_after_exceeds,
    wait_for_retry_after,
)


def _rate_limited(headers: dict) -> EmbeddingException:
    """A 429 wrapped the way the engines re-raise provider failures."""
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "http://api.test")
    )
    provider_error = openai.RateLimitError("rate limited", response=response, body=None)
    try:
        raise EmbeddingException("embedding failed") from provider_error
    except EmbeddingException as wrapped:
        return wrapped


def _state(error: BaseException, seconds_since_start: float = 0.0):
    outcome = SimpleNamespace(failed=True, exception=lambda: error)
    return SimpleNamespace(outcome=outcome, seconds_since_start=seconds_since_start)


def _fallback(_retry_state) -> float:
    return 42.0


def test_delta_seconds_retry_after_sets_the_wait():
    wait = wait_for_retry_after(_fallback)
    assert wait(_state(_rate_limited({"Retry-After": "5"}))) == 5.0


def test_http_date_retry_after_sets_the_wait():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    wait = wait_for_retry_after(_fallback)
    seconds = wait(_state(_rate_limited({"Retry-After": format_datetime(when, usegmt=True)})))
    assert 25.0 <= seconds <= 30.0


def test_without_retry_after_the_backoff_applies_and_nothing_stops():
    state = _state(_rate_limited({}))
    assert wait_for_retry_after(_fallback)(state) == 42.0
    assert stop_when_retry_after_exceeds(128)(state) is False


def test_retry_after_beyond_the_window_stops_instead_of_sleeping():
    stop = stop_when_retry_after_exceeds(128)
    assert stop(_state(_rate_limited({"Retry-After": "60"}), seconds_since_start=100)) is True
    assert stop(_state(_rate_limited({"Retry-After": "20"}), seconds_since_start=100)) is False


def test_ollama_error_carries_its_retry_after():
    from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
        OllamaEmbeddingAPIError,
    )

    error = OllamaEmbeddingAPIError("busy", status_code=429, retry_after="3")
    assert wait_for_retry_after(_fallback)(_state(error)) == 3.0

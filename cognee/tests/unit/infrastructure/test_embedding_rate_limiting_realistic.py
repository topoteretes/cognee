"""Embedding rate limiting: the client-side limiter paces requests, it never rejects them.

``embedding_rate_limiter_context_manager`` returns an ``aiolimiter.AsyncLimiter``
(a leaky bucket) when ``EMBEDDING_RATE_LIMIT_ENABLED`` is set. Entering it waits
for capacity, so requests above the configured rate are delayed until they fit
the budget instead of failing. These tests pin that contract: every request
succeeds, and a burst takes as long as the configured rate requires.
"""

import asyncio
import time

import pytest

from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_config,
)
from cognee.shared import rate_limiting
from cognee.tests.unit.infrastructure.mock_embedding_engine import MockEmbeddingEngine


@pytest.fixture
def embedding_rate_limit(monkeypatch):
    """Configure the embedding limiter for one test, with fresh config and limiter caches."""

    def _configure(enabled: bool, requests: int = 3, interval: int = 1) -> None:
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_ENABLED", "true" if enabled else "false")
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_REQUESTS", str(requests))
        monkeypatch.setenv("EMBEDDING_RATE_LIMIT_INTERVAL", str(interval))
        monkeypatch.setenv("MOCK_EMBEDDING", "true")
        get_embedding_config.cache_clear()
        rate_limiting._embedding_rate_limiter = None

    yield _configure

    get_embedding_config.cache_clear()
    rate_limiting._embedding_rate_limiter = None


async def _burst(engine: MockEmbeddingEngine, size: int) -> tuple[list, float]:
    """Send ``size`` concurrent requests; return their results and the elapsed wall time."""
    start = time.monotonic()
    results = await asyncio.gather(
        *(engine.embed_text([f"text {i}"]) for i in range(size)), return_exceptions=True
    )
    return results, time.monotonic() - start


@pytest.mark.asyncio
async def test_embedding_rate_limiter_paces_a_burst_instead_of_rejecting(embedding_rate_limit):
    """9 concurrent requests at 3 per second: all succeed, and the burst takes ~2s."""
    embedding_rate_limit(enabled=True, requests=3, interval=1)
    engine = MockEmbeddingEngine(dimensions=4)

    results, elapsed = await _burst(engine, 9)

    # Nothing is rejected: the limiter delays requests, it never raises.
    assert all(not isinstance(result, BaseException) for result in results), results
    assert all(result == [[0.1] * 4] for result in results)
    # A leaky bucket of 3 admits 3 at once, then drains at 3/s, so the other 6
    # need about 2s. The lower bound is what proves pacing happened.
    assert elapsed >= 1.5, f"burst finished in {elapsed:.2f}s; the limiter did not pace it"


@pytest.mark.asyncio
async def test_disabled_embedding_rate_limiter_does_not_pace(embedding_rate_limit):
    """The same burst with the limiter off completes immediately (the control case)."""
    embedding_rate_limit(enabled=False)
    engine = MockEmbeddingEngine(dimensions=4)

    results, elapsed = await _burst(engine, 9)

    assert all(not isinstance(result, BaseException) for result in results), results
    assert elapsed < 1.0, f"burst took {elapsed:.2f}s with rate limiting disabled"


@pytest.mark.asyncio
async def test_mock_failures_surface_through_the_rate_limited_engine(embedding_rate_limit):
    """Failures raised by the engine propagate; the limiter does not swallow or mask them."""
    embedding_rate_limit(enabled=True, requests=10, interval=1)
    engine = MockEmbeddingEngine(dimensions=4)
    engine.configure_mock(fail_every_n_requests=3)

    outcomes = []
    for i in range(9):
        try:
            await engine.embed_text([f"text {i}"])
            outcomes.append("ok")
        except RuntimeError:
            outcomes.append("failed")

    assert outcomes == ["ok", "ok", "failed"] * 3


def test_embedding_rate_limit_fields_on_embedding_config(monkeypatch):
    """The embedding rate-limit knobs live on EmbeddingConfig and read their env vars."""
    monkeypatch.setenv("EMBEDDING_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_RATE_LIMIT_REQUESTS", "7")
    get_embedding_config.cache_clear()
    try:
        cfg = get_embedding_config()
        assert cfg.embedding_rate_limit_enabled is True
        assert cfg.embedding_rate_limit_requests == 7
    finally:
        get_embedding_config.cache_clear()


def test_llm_config_no_longer_defines_embedding_rate_limit_fields():
    """The knobs were moved out of LLMConfig, leaving a single source of truth."""
    from cognee.infrastructure.llm.config import LLMConfig

    for field in (
        "embedding_rate_limit_enabled",
        "embedding_rate_limit_requests",
        "embedding_rate_limit_interval",
        "embedding_rate_limit_tokens",
    ):
        assert field not in LLMConfig.model_fields

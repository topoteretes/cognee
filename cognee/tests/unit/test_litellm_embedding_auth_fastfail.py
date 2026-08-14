"""Regression test: the embedding engine must NOT retry on auth failures.

Rationale: prior to the fix, the tenacity ``retry`` decorator only
skipped ``NotFoundError`` and ``CancelledError``. On an invalid
``LLM_API_KEY`` litellm raised ``AuthenticationError`` and the engine
walked the full 8s -> 128s exponential backoff ladder, hanging the CLI
for two full minutes before surfacing the failure. This test pins the
retry set so the fast-fail behaviour cannot silently regress.
"""

import asyncio

import httpx
import litellm
import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


def _fake_response(status_code: int) -> httpx.Response:
    """Build a minimal ``httpx.Response`` for litellm exception constructors.

    litellm.PermissionDeniedError requires a real Response object, so a
    plain string is not enough. Only the status code is meaningful here.
    """
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://example.invalid"),
    )


@pytest.mark.asyncio
async def test_auth_error_bypasses_retry(monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_auth(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.AuthenticationError(
            message="invalid api key",
            llm_provider="openai",
            model="text-embedding-3-large",
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_auth)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await engine.embed_text(["hello world"])

    # The critical assertion: exactly one attempt, no backoff ladder.
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_permission_denied_bypasses_retry(monkeypatch):
    # A 403 PermissionDeniedError is terminal (org/region/model access denied).
    # Note: quota exhaustion surfaces as a 429 RateLimitError, which is
    # transient and intentionally still retried — so it is not tested here.
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_perm(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.PermissionDeniedError(
            message="You do not have access to model text-embedding-3-large",
            llm_provider="openai",
            model="text-embedding-3-large",
            response=_fake_response(403),
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_perm)

    with pytest.raises(litellm.exceptions.PermissionDeniedError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_cancelled_error_bypasses_retry(monkeypatch):
    """Existing behaviour, kept as a guard rail."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_cancelled(**kwargs):
        calls["count"] += 1
        raise asyncio.CancelledError()

    monkeypatch.setattr(litellm, "aembedding", _raise_cancelled)

    with pytest.raises(asyncio.CancelledError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1

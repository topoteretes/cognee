"""Regression tests: the embedding engine must NOT retry configuration errors.

Rationale (SDK-810): terminal, deterministic failures used to run the full
``stop_after_delay(128)`` backoff ladder and then surface as a generic 422:

* ``litellm.NotFoundError`` (mis-typed EMBEDDING_MODEL) was re-raised wrapped
  in ``EmbeddingException``, defeating the retry decorator's exclusion of the
  litellm class;
* the catch-all wrapped EVERY terminal error (missing ``transformers``
  package, HuggingFace repo resolution failures, non-transient 4xx) into the
  retryable ``EmbeddingException``.

These tests pin the fast-fail behaviour: exactly one attempt, and the real
cause visible in the surfaced error. The complementary direction — a
transient 429 KEEPS its ladder — is pinned by
``test_transient_rate_limit_is_still_retried`` in
``test_embedding_budget_fastfail.py``.
"""

import httpx
import litellm
import pytest

from cognee.infrastructure.databases.exceptions import EmbeddingConfigurationError
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


def _fake_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://example.invalid"),
    )


@pytest.mark.asyncio
async def test_not_found_error_bypasses_retry(monkeypatch):
    """A 404 (model does not exist) is terminal: one attempt, litellm class intact."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_not_found(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.NotFoundError(
            message="The model `text-embeddin-3-large` does not exist",
            llm_provider="openai",
            model="text-embeddin-3-large",
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_not_found)

    # The original litellm class must propagate unwrapped: that is both what
    # lets the retry decorator exclude it and what the CLI remediation table
    # matches on ("litellm.notfounderror").
    with pytest.raises(litellm.exceptions.NotFoundError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_plain_bad_request_bypasses_retry(monkeypatch):
    """A non-length 400 is a deterministic client error: one attempt, no ladder."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_bad_request(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.BadRequestError(
            message="Unknown parameter: 'dimensions'",
            llm_provider="openai",
            model="text-embedding-3-large",
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_bad_request)

    with pytest.raises(litellm.exceptions.BadRequestError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_missing_package_bypasses_retry_and_names_the_cause(monkeypatch):
    """A missing optional package is terminal and must surface the real cause.

    Before the fix a ``ModuleNotFoundError`` retried for 128s and then surfaced
    as "Embedding failed due to an unexpected error ... (Status code: 422)",
    which users misread as a provider 422 response.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_missing_module(**kwargs):
        calls["count"] += 1
        raise ModuleNotFoundError("No module named 'transformers'", name="transformers")

    monkeypatch.setattr(litellm, "aembedding", _raise_missing_module)

    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1
    # The message embeds ``TypeName: text`` so the real cause survives the 422.
    assert "ModuleNotFoundError" in str(exc_info.value)
    assert "transformers" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wrapped_terminal_4xx_bypasses_retry(monkeypatch):
    """A non-408/429 4xx hidden inside a wrapper is classified via the cause chain."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_wrapped_422(**kwargs):
        calls["count"] += 1
        try:
            raise litellm.exceptions.UnprocessableEntityError(
                message="Input failed provider validation",
                llm_provider="openai",
                model="text-embedding-3-large",
                response=_fake_response(422),
            )
        except litellm.exceptions.UnprocessableEntityError as inner:
            raise RuntimeError("wrapper hides the class") from inner

    monkeypatch.setattr(litellm, "aembedding", _raise_wrapped_422)

    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1
    assert "UnprocessableEntityError" in str(exc_info.value)


@pytest.mark.asyncio
async def test_preflight_surfaces_real_error_instead_of_timeout(monkeypatch):
    """The connection preflight must report the real 404, not burn its 30s window.

    ``test_embedding_connection`` used to call the retry-decorated
    ``embed_text``, so a bad EMBEDDING_MODEL spent the whole preflight timeout
    inside the backoff ladder and was misreported as "endpoint unreachable".
    """
    from types import SimpleNamespace

    import cognee.infrastructure.databases.vector as vector_module
    from cognee.infrastructure.llm.utils import test_embedding_connection

    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_not_found(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.NotFoundError(
            message="The model `text-embeddin-3-large` does not exist",
            llm_provider="openai",
            model="text-embeddin-3-large",
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_not_found)

    async def _fake_get_vector_engine_async():
        return SimpleNamespace(embedding_engine=engine)

    monkeypatch.setattr(vector_module, "get_vector_engine_async", _fake_get_vector_engine_async)

    with pytest.raises(litellm.exceptions.NotFoundError):
        await test_embedding_connection()

    # Exactly one undecorated attempt: no ladder, no fake timeout.
    assert calls["count"] == 1

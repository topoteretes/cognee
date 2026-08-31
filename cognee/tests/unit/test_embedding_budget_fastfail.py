"""Regression test: the embedding path must NOT retry a budget rejection.

Rationale (COG-6475): a LiteLLM spend cap cannot clear inside a retry window,
but the embedding engines classified terminal errors by exception class only, so
a budget rejection ran the full ``stop_after_delay(128)`` ladder -- 7 attempts,
~130s -- for every doomed batch. Embedding batches are gated by a semaphore of
``embedding_max_concurrent_data_points // batch_size`` (4 by default), so each
doomed batch also holds a slot for that whole time.

Class-based exclusion cannot fix this, which is why these tests exercise the
engines rather than a type tuple:

* ``litellm.BudgetExceededError`` subclasses plain ``Exception``;
* against a LiteLLM proxy the client receives an ordinary ``RateLimitError``
  instead of that class;
* the engines re-raise provider failures as ``EmbeddingException(...) from
  error``, so by the time tenacity sees the failure the provider class is gone.

Both failure directions matter. A false negative resurrects the ladder; a false
positive turns a genuinely transient rate limit into a hard 402, which is why
the transient cases below are pinned just as tightly.
"""

import asyncio

import aiohttp
import httpx
import litellm
import openai
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
    OpenAICompatibleEmbeddingEngine,
)
from cognee.infrastructure.databases.exceptions import (
    EmbeddingCredentialsError,
    EmbeddingException,
)
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

BUDGET_MESSAGE = "Budget has been exceeded! Current cost: 20.00066499999998, Max budget: 20.0"


def _proxy_budget_error(message: str = BUDGET_MESSAGE) -> litellm.RateLimitError:
    """The client-side error litellm raises for a LiteLLM-proxy spend-cap 429.

    The proxy raises ``BudgetExceededError`` server-side; the client only ever
    sees the status mapped back onto a generic ``RateLimitError``.
    """
    return litellm.RateLimitError(
        message=f"litellm.RateLimitError: RateLimitError: Litellm_proxyException - {message}",
        llm_provider="openai",
        model="litellm_proxy/litellm",
    )


def _openai_structured_budget_error() -> openai.RateLimitError:
    """A proxy 429 whose body is still readable: the structured signal.

    litellm's own client consumes the response body before cognee sees the
    error, so the LLM side can only ever match on message text. Through the
    openai SDK the body survives and the proxy's ``error.type`` is the more
    robust signal, so the message here deliberately omits the budget sentence:
    only the structured check can classify this one.
    """
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "http://localhost:8099/v1/embeddings"),
        json={
            "error": {
                "message": "Rejected by proxy",
                "type": "budget_exceeded",
                "code": "429",
            }
        },
    )
    return openai.RateLimitError("Rejected by proxy", response=response, body=None)


def _openai_transient_rate_limit() -> openai.RateLimitError:
    """A per-minute rate limit: readable body, but not a spend cap."""
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "http://localhost:8099/v1/embeddings"),
        json={"error": {"message": "Rate limit reached", "type": "rate_limit_error"}},
    )
    return openai.RateLimitError("Rate limit reached", response=response, body=None)


def _openai_auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "invalid api key",
        response=httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "http://localhost:8099/v1/embeddings"),
        ),
        body=None,
    )


def _auth_error() -> litellm.exceptions.AuthenticationError:
    """A terminal error used to stop a ladder that is expected to run."""
    return litellm.exceptions.AuthenticationError(
        message="invalid api key",
        llm_provider="openai",
        model="text-embedding-3-large",
    )


@pytest.mark.asyncio
async def test_litellm_budget_error_bypasses_retry(monkeypatch):
    """litellm's own budget error: one attempt, surfaced as the actionable 402."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_budget(**kwargs):
        calls["count"] += 1
        raise litellm.BudgetExceededError(current_cost=20.0, max_budget=20.0)

    monkeypatch.setattr(litellm, "aembedding", _raise_budget)

    with pytest.raises(LLMPaymentRequiredError):
        await engine.embed_text(["hello world"])

    # The critical assertion: exactly one attempt, no backoff ladder.
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_litellm_proxy_budget_rejection_bypasses_retry(monkeypatch):
    """The shape production actually produces: a 429 carrying the budget sentence."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_proxy_budget(**kwargs):
        calls["count"] += 1
        raise _proxy_budget_error()

    monkeypatch.setattr(litellm, "aembedding", _raise_proxy_budget)

    with pytest.raises(LLMPaymentRequiredError) as exc_info:
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1
    # The provider's own sentence is carried through, so the 402 is actionable.
    assert "Budget has been exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_litellm_budget_rejection_as_bad_request_bypasses_retry(monkeypatch):
    """A budget rejection arriving as a 400 gets the same 402 as every other shape.

    The proxy's transport status must not decide the API error family: this is the
    same spend cap as the 429 above, so the caller must not have to handle two
    types for it. The ``BadRequestError`` clause catches this before the handler
    that converts, so the conversion has to happen inside that clause.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_bad_request_budget(**kwargs):
        calls["count"] += 1
        raise litellm.exceptions.BadRequestError(
            message=f"Litellm_proxyException - {BUDGET_MESSAGE}",
            llm_provider="openai",
            model="litellm_proxy/litellm",
        )

    monkeypatch.setattr(litellm, "aembedding", _raise_bad_request_budget)

    with pytest.raises(LLMPaymentRequiredError) as exc_info:
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1
    assert exc_info.value.status_code == 402
    assert "Budget has been exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transient_rate_limit_is_still_retried(monkeypatch):
    """A per-minute rate limit is transient and must keep its ladder.

    The second attempt raises a terminal auth error purely to stop the ladder
    after one backoff instead of running the full 128s.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_transient_then_terminal(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise litellm.RateLimitError(
                message="litellm.RateLimitError: Rate limit reached for gpt-4o",
                llm_provider="openai",
                model="text-embedding-3-large",
            )
        raise _auth_error()

    monkeypatch.setattr(litellm, "aembedding", _raise_transient_then_terminal)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_transient_error_mentioning_a_budget_is_still_retried(monkeypatch):
    """False-positive guard: prose about a budget is not a proxy rejection.

    A provider message can mention an exceeded budget without being a spend-cap
    rejection, so this stays retryable: only the provider's whole sentence
    counts. Text that does carry that whole sentence is classified as a budget
    error, which is an accepted limitation pinned by ``test_budget_exhaustion``
    rather than something this test covers.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    engine = LiteLLMEmbeddingEngine(dimensions=4)

    calls = {"count": 0}

    async def _raise_innocent_then_terminal(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise litellm.RateLimitError(
                message=(
                    "litellm.RateLimitError: Rate limit reached. Context: "
                    "'The marketing budget has been exceeded! Please review before Friday.'"
                ),
                llm_provider="openai",
                model="text-embedding-3-large",
            )
        raise _auth_error()

    monkeypatch.setattr(litellm, "aembedding", _raise_innocent_then_terminal)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_openai_compatible_budget_rejection_bypasses_retry(monkeypatch):
    """The same guarantee for the engine that talks to any OpenAI-compatible URL.

    A LiteLLM proxy is a valid ``EMBEDDING_ENDPOINT`` for this engine too, so it
    can see the identical rejection.
    """
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=4,
        endpoint="http://localhost:8099",
        api_key="test-key",
    )

    engine._client = MagicMock()
    engine._client.embeddings.create = AsyncMock(side_effect=_proxy_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await engine.embed_text(["hello world"])

    assert engine._client.embeddings.create.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_error_still_bypasses_retry(monkeypatch):
    """Existing behaviour, kept as a guard rail against the predicate swap."""
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


@pytest.mark.asyncio
async def test_openai_compatible_structured_budget_rejection_bypasses_retry():
    """The structured ``error.type`` signal alone must be enough.

    A proxy that changes its wording still sets ``type: budget_exceeded``, and
    through the openai SDK the body is readable, so this path must not depend on
    the message regex.
    """
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=4,
        endpoint="http://localhost:8099",
        api_key="test-key",
    )

    engine._client = MagicMock()
    engine._client.embeddings.create = AsyncMock(side_effect=_openai_structured_budget_error())

    with pytest.raises(LLMPaymentRequiredError):
        await engine.embed_text(["hello world"])

    assert engine._client.embeddings.create.await_count == 1


@pytest.mark.asyncio
async def test_openai_compatible_transient_rate_limit_is_still_retried():
    """The transient mirror for the engine whose conversion sits highest.

    ``raise_if_budget_exhausted`` runs ahead of this engine's context-window
    split recovery, so a false positive here would be the most damaging one in
    the change. The second attempt raises an auth error, which is terminal only
    because it is re-raised unwrapped, so this also pins that.
    """
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=4,
        endpoint="http://localhost:8099",
        api_key="test-key",
    )

    calls = {"count": 0}

    async def _transient_then_terminal(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _openai_transient_rate_limit()
        raise _openai_auth_error()

    engine._client = MagicMock()
    engine._client.embeddings.create = AsyncMock(side_effect=_transient_then_terminal)

    with pytest.raises(EmbeddingCredentialsError):
        await engine.embed_text(["hello world"])

    # A callable, not a two-item list: if the auth branch regressed, call 3+
    # keeps raising the auth error instead of StopAsyncIteration, so the failure
    # names the real cause instead of running the full ladder first.
    assert calls["count"] == 2


class _FakeAiohttpResponse:
    """Minimal stand-in for the aiohttp response ``_get_embedding`` reads."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self) -> "_FakeAiohttpResponse":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


@pytest.mark.asyncio
async def test_ollama_engine_budget_rejection_bypasses_retry(monkeypatch):
    """``EMBEDDING_PROVIDER=ollama`` can be pointed at a LiteLLM proxy.

    The endpoint is never validated for provider shape and a proxy's
    OpenAI-shaped success body is accepted by the ``"data"`` branch, so this
    engine works against a proxy until the spend cap. The rejection then
    arrives as a ``RuntimeError`` built from the error body, which is why the
    classification cannot be a type tuple here either.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine."
        "OllamaEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
            OllamaEmbeddingEngine,
        )

        engine = OllamaEmbeddingEngine(model="test-model", dimensions=4)

    calls = {"count": 0}
    proxy_body = {
        "error": {
            "message": f"litellm.BudgetExceededError: {BUDGET_MESSAGE}",
            "type": "budget_exceeded",
            "code": "429",
        }
    }

    def _fake_post(self, *args, **kwargs):
        calls["count"] += 1
        return _FakeAiohttpResponse(proxy_body)

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    with pytest.raises(LLMPaymentRequiredError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_ollama_engine_object_shaped_context_window_error_is_terminal(monkeypatch):
    """An over-length error in an object-shaped body must not be retried.

    The membership tests in ``_get_embedding`` read the error body directly, so
    an OpenAI-compatible proxy's object body used to test dict keys, always
    False, and a terminal over-length error became a retryable RuntimeError that
    burned the full ladder before ``embed_text`` could split. Reachable through
    the same misconfiguration that makes the budget case above reachable.
    """
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch(
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine."
        "OllamaEmbeddingEngine.get_tokenizer",
        return_value=Mock(),
    ):
        from cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine import (
            OllamaEmbeddingEngine,
        )

        engine = OllamaEmbeddingEngine(model="test-model", dimensions=4)

    calls = {"count": 0}
    proxy_body = {
        "error": {
            "message": "This model's maximum context length is 8192 tokens",
            "type": "invalid_request_error",
        }
    }

    def _fake_post(self, *args, **kwargs):
        calls["count"] += 1
        return _FakeAiohttpResponse(proxy_body)

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    # "ab" cannot be split further, so the recovery path gives up immediately and
    # the call count is the whole assertion: one attempt, no ladder.
    with pytest.raises(EmbeddingException, match="too short to split further"):
        await engine.embed_text(["ab"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_openai_compatible_credentials_error_is_terminal_and_stays_a_422():
    """A rejected key must fail fast without becoming a 500.

    Re-raising the openai class bare would fast-fail too, but it leaves the
    ``CogneeApiError`` family, so a router's ``except Exception`` turns a
    configuration mistake into "Internal server error". The status code is as
    much part of this contract as the attempt count.
    """
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=4,
        endpoint="http://localhost:8099",
        api_key="wrong-key",
    )

    engine._client = MagicMock()
    engine._client.embeddings.create = AsyncMock(side_effect=_openai_auth_error())

    with pytest.raises(EmbeddingCredentialsError) as exc_info:
        await engine.embed_text(["hello world"])

    assert engine._client.embeddings.create.await_count == 1
    assert exc_info.value.status_code == 422
    # The provider's own text is what makes the 422 actionable.
    assert "invalid api key" in exc_info.value.message

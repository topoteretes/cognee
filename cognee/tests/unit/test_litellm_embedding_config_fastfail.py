"""Regression tests: the LiteLLM embedding engine must not retry terminal errors.

Rationale (SDK-810): a mis-typed EMBEDDING_MODEL, a plain 400/422 or a missing
package ran the full ``stop_after_delay(128)`` backoff ladder, because the
handlers wrapped them in the retryable ``EmbeddingException`` before tenacity
could match the exclusion list. The failure then surfaced with a generic
message next to the 422 status, which read like a provider response.

The transient direction (a 429 keeps its ladder) is pinned by
``test_transient_rate_limit_is_still_retried`` in ``test_embedding_budget_fastfail.py``.
"""

import inspect
from types import SimpleNamespace

import httpx
import litellm
import pytest

from cognee.infrastructure.databases.exceptions import EmbeddingException
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("POST", "https://example.invalid"))


def _not_found() -> litellm.exceptions.NotFoundError:
    return litellm.exceptions.NotFoundError(
        message="The model `text-embeddin-3-large` does not exist",
        llm_provider="openai",
        model="text-embeddin-3-large",
    )


def _auth_error() -> litellm.exceptions.AuthenticationError:
    return litellm.exceptions.AuthenticationError(
        message="Incorrect API key provided", llm_provider="openai", model="text-embedding-3-large"
    )


def _engine(monkeypatch, side_effect):
    """An engine whose litellm call runs ``side_effect(call_number)``; returns (engine, calls)."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    calls = {"count": 0}

    async def _aembedding(**kwargs):
        calls["count"] += 1
        raise side_effect(calls["count"])

    monkeypatch.setattr(litellm, "aembedding", _aembedding)
    return LiteLLMEmbeddingEngine(dimensions=4), calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _not_found(),
        litellm.exceptions.BadRequestError(
            message="Unknown parameter: 'dimensions'",
            llm_provider="openai",
            model="text-embedding-3-large",
        ),
        litellm.exceptions.UnprocessableEntityError(
            message="Input failed provider validation",
            llm_provider="openai",
            model="text-embedding-3-large",
            response=_response(422),
        ),
    ],
    ids=["not_found_404", "bad_request_400", "unprocessable_422"],
)
async def test_terminal_error_makes_one_attempt_and_keeps_its_class(monkeypatch, error):
    engine, calls = _engine(monkeypatch, lambda _: error)

    with pytest.raises(type(error)):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


def _litellm_missing_sdk() -> litellm.exceptions.APIConnectionError:
    """What litellm raises for a bedrock model without boto3 (checked against litellm).

    litellm raises its APIConnectionError (status 500) while handling the ImportError, so
    the ImportError is the error's ``__context__``; ``__cause__`` is None.
    """
    try:
        raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
    except ImportError:
        try:
            raise litellm.exceptions.APIConnectionError(
                message="litellm.APIConnectionError: Missing boto3 to call bedrock.",
                llm_provider="bedrock",
                model="amazon.titan-embed-text-v2:0",
            )
        except litellm.exceptions.APIConnectionError as error:
            return error


@pytest.mark.asyncio
async def test_missing_provider_sdk_behind_litellm_makes_one_attempt(monkeypatch):
    """A missing provider SDK is terminal even though litellm reports it as a 500."""
    wrapped = _litellm_missing_sdk()
    assert isinstance(wrapped.__context__, ImportError) and wrapped.__cause__ is None
    engine, calls = _engine(monkeypatch, lambda _: wrapped)

    with pytest.raises(EmbeddingException, match="boto3"):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_missing_package_raised_directly_makes_one_attempt(monkeypatch):
    engine, calls = _engine(
        monkeypatch, lambda _: ModuleNotFoundError("No module named 'boto3'", name="boto3")
    )

    with pytest.raises(EmbeddingException, match="boto3"):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_connection_error_without_a_missing_package_is_still_retried(monkeypatch):
    """Only an ImportError behind litellm's APIConnectionError is terminal, not the class itself."""

    def _connection_reset_then_auth(call: int) -> Exception:
        if call == 1:
            return litellm.exceptions.APIConnectionError(
                message="Connection reset by peer",
                llm_provider="openai",
                model="text-embedding-3-large",
            )
        return _auth_error()

    engine, calls = _engine(monkeypatch, _connection_reset_then_auth)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_server_error_is_still_retried(monkeypatch):
    """A 5xx is transient and keeps its ladder (the auth error only stops it early)."""

    def _server_error_then_auth(call: int) -> Exception:
        if call == 1:
            return litellm.exceptions.InternalServerError(
                message="upstream hiccup", llm_provider="openai", model="text-embedding-3-large"
            )
        return _auth_error()

    engine, calls = _engine(monkeypatch, _server_error_then_auth)

    with pytest.raises(litellm.exceptions.AuthenticationError):
        await engine.embed_text(["hello world"])

    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_wrapped_error_message_names_the_real_cause(monkeypatch):
    """The generic wrap must carry the provider error, not just a bare 422."""
    engine, _ = _engine(monkeypatch, lambda _: RuntimeError("provider said no"))
    # One attempt without the retry decorator: the message is what is under test.
    embed_once = inspect.unwrap(LiteLLMEmbeddingEngine.embed_text)

    with pytest.raises(EmbeddingException) as exc_info:
        await embed_once(engine, ["hello world"])

    assert "RuntimeError: provider said no" in str(exc_info.value)


@pytest.mark.asyncio
async def test_preflight_surfaces_real_404_in_one_attempt(monkeypatch):
    """The connection preflight reports the real 404 instead of timing out in the ladder."""
    import cognee.infrastructure.databases.vector as vector_module
    from cognee.infrastructure.llm.utils import test_embedding_connection

    engine, calls = _engine(monkeypatch, lambda _: _not_found())

    async def _fake_get_vector_engine_async():
        return SimpleNamespace(embedding_engine=engine)

    monkeypatch.setattr(vector_module, "get_vector_engine_async", _fake_get_vector_engine_async)

    with pytest.raises(litellm.exceptions.NotFoundError):
        await test_embedding_connection()

    assert calls["count"] == 1

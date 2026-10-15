import base64
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


@pytest.mark.asyncio
async def test_embed_text_filters_invalid_inputs(monkeypatch):
    # Use monkeypatch so env is restored automatically
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    # Float JSON, so the stub below can answer with plain lists.
    engine = LiteLLMEmbeddingEngine(dimensions=4, encoding_format="float")

    # Stub litellm call to return distinct vectors for each valid input
    UNIQUE_A = [1.0, 1.0, 1.0, 1.0]
    UNIQUE_B = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_C = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_D = [2.0, 2.0, 2.0, 2.0]
    UNIQUE_E = [2.0, 2.0, 2.0, 2.0]

    class _Resp:
        def __init__(self, data):
            self.data = data

    async def fake_aembedding(**kwargs):
        # kwargs["input"] should already be filtered by engine
        # In our inputs below, valid entries are "valid" and "ok!"
        return _Resp(
            [
                {"embedding": UNIQUE_A},
                {"embedding": UNIQUE_B},
                {"embedding": UNIQUE_C},
                {"embedding": UNIQUE_D},
                {"embedding": UNIQUE_E},
            ]
        )

    # Patch the litellm call used in the engine module
    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    inputs = ["", "(", "valid", "   ", "ok!"]
    result = await engine.embed_text(inputs)

    # Output length must match input length
    assert len(result) == len(inputs)

    # Invalid entries should be zero vectors
    assert result[0] == [0.0] * 4
    assert result[3] == [0.0] * 4

    # Valid entries must map to the correct positions
    assert result[1] == UNIQUE_B
    assert result[2] == UNIQUE_C
    assert result[4] == UNIQUE_E


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model", "endpoint", "configured", "expected_encoding_format"),
    [
        # Documented OpenRouter config (issue #3660): "openrouter/" model prefix.
        (
            "custom",
            "openrouter/openai/text-embedding-3-small",
            "https://example.invalid/v1",
            None,
            "float",
        ),
        # Explicit provider name.
        ("openrouter", "text-embedding-3-small", "https://example.invalid/v1", None, "float"),
        # Prefix detection is case-insensitive.
        (
            "custom",
            "OpenRouter/openai/text-embedding-3-small",
            "https://example.invalid/v1",
            None,
            "float",
        ),
        # Custom endpoint aimed at openrouter.ai with an unprefixed model: litellm
        # drives this through its OpenAI handler (the branch that injected the null),
        # so the guard must still fire based on the endpoint host.
        (
            "custom",
            "openai/text-embedding-3-small",
            "https://openrouter.ai/api/v1",
            None,
            "float",
        ),
        # OpenAI's own API asks for base64 vectors.
        ("openai", "openai/text-embedding-3-large", None, None, "base64"),
        ("openai", "text-embedding-3-large", None, None, "base64"),
        # Every other route sends the request it always did.
        ("openai", "openai/text-embedding-3-large", "https://proxy.invalid/v1", None, None),
        ("azure", "azure/my-embedding-deployment", "https://example.invalid", None, None),
        ("gemini", "gemini/text-embedding-004", "https://example.invalid/v1", None, None),
        # EMBEDDING_ENCODING_FORMAT opts a route in or out.
        ("openai", "openai/text-embedding-3-large", "https://proxy.invalid/v1", "base64", "base64"),
        ("openrouter", "text-embedding-3-small", "https://example.invalid/v1", "base64", "base64"),
        ("openai", "openai/text-embedding-3-large", None, "float", None),
    ],
)
async def test_embed_text_sets_encoding_format_per_route(
    monkeypatch, provider, model, endpoint, configured, expected_encoding_format
):
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider=provider,
            model=model,
            dimensions=2,
            api_key="test-key",
            endpoint=endpoint,
            encoding_format=configured,
        )

    # Answer in the format the route asked for; both values are exact in float32.
    vector = [0.5, 0.25]
    payload = (
        base64.b64encode(struct.pack("<2f", *vector)).decode()
        if engine.encoding_format == "base64"
        else vector
    )
    response = SimpleNamespace(data=[{"embedding": payload}])
    mock_aembedding = AsyncMock(return_value=response)

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", mock_aembedding)

    result = await engine.embed_text(["hello"])

    expected_kwargs = {
        "model": model,
        "input": ["hello"],
        "api_key": "test-key",
        "api_base": endpoint,
        "api_version": None,
        "dimensions": 2,
    }
    if expected_encoding_format is not None:
        expected_kwargs["encoding_format"] = expected_encoding_format

    mock_aembedding.assert_awaited_once_with(**expected_kwargs)
    assert result == [vector]


@pytest.mark.asyncio
async def test_embed_text_decodes_base64_vectors(monkeypatch):
    """A base64 payload decodes to the float32 values it carries; float lists pass through."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider="openai",
            model="openai/text-embedding-3-large",
            dimensions=4,
            api_key="test-key",
        )

    vector = [0.0123, -0.5, 0.25, 1e-7]
    float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()
    response = SimpleNamespace(data=[{"embedding": encoded}, {"embedding": encoded}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(return_value=response))

    result = await engine.embed_text(["a", "b"])

    assert result == [float32_vector, float32_vector]
    assert all(isinstance(v, float) for v in result[0])


@pytest.mark.asyncio
async def test_float_list_on_a_base64_route_fails_without_retry(monkeypatch):
    """A proxy that translates base64 for Bedrock answers with a list of 0/1: raise, once."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="openai/titan",
            dimensions=4,
            api_key="test-key",
            endpoint="http://litellm-proxy:4000",
            encoding_format="base64",
        )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    response = SimpleNamespace(data=[{"embedding": [1.0, 0.0, 1.0, 1.0]}])
    aembedding = AsyncMock(return_value=response)
    monkeypatch.setattr(mod.litellm, "aembedding", aembedding)

    with pytest.raises(mod.EmbeddingEncodingError, match="EMBEDDING_ENCODING_FORMAT"):
        await engine.embed_text(["hello"])

    assert aembedding.await_count == 1


@pytest.mark.parametrize(
    "message",
    [
        # pydantic 2.10
        (
            "Pydantic serializer warnings:\n  Expected `list[float]` but got `str` with value "
            "`'AAAA'` - serialized value may not be as expected"
        ),
        # pydantic 2.11
        (
            "Pydantic serializer warnings:\n  PydanticSerializationUnexpectedValue(Expected "
            "`list[float]` - serialized value may not be as expected [input_value='AAAA', "
            "input_type=str])"
        ),
        # pydantic 2.12+
        (
            "Pydantic serializer warnings:\n  PydanticSerializationUnexpectedValue(Expected "
            "`list[float]` - serialized value may not be as expected [field_name='embedding', "
            "input_value='AAAA', input_type=str])"
        ),
    ],
)
def test_base64_serializer_warning_is_filtered_on_every_pydantic(message):
    import re

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    assert re.match(mod._BASE64_SERIALIZER_WARNING, message)
    # Any other value type, or any other field type, is still shown.
    other_type = message.replace("but got `str`", "but got `dict`").replace(
        "input_type=str", "input_type=dict"
    )
    assert not re.match(mod._BASE64_SERIALIZER_WARNING, other_type)
    assert not re.match(mod._BASE64_SERIALIZER_WARNING, message.replace("list[float]", "int"))


@pytest.mark.asyncio
async def test_rejected_base64_logs_the_setting_to_change(monkeypatch):
    """A 400 while base64 is requested names EMBEDDING_ENCODING_FORMAT and still raises."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="openai/m",
            dimensions=2,
            api_key="test-key",
            endpoint="http://localhost:8000/v1",
            encoding_format="base64",
        )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    error = mod.litellm.exceptions.BadRequestError(
        message="unsupported encoding_format", model="openai/m", llm_provider="openai"
    )
    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(side_effect=error))
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(mod.litellm.exceptions.BadRequestError):
        await engine.embed_text(["hello"])

    logged.assert_called_with(mod.ENCODING_FORMAT_HINT)


@pytest.mark.asyncio
async def test_unrelated_400_does_not_blame_the_encoding(monkeypatch):
    """With base64 set, a 400 that is not about the encoding gets no hint."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="openai/m",
            dimensions=2,
            api_key="test-key",
            endpoint="http://localhost:8000/v1",
            encoding_format="base64",
        )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    error = mod.litellm.exceptions.BadRequestError(
        message="This model does not support specifying dimensions.",
        model="openai/m",
        llm_provider="openai",
    )
    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(side_effect=error))
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(mod.litellm.exceptions.BadRequestError):
        await engine.embed_text(["hello"])

    assert all(call.args[:1] != (mod.ENCODING_FORMAT_HINT,) for call in logged.call_args_list)


@pytest.mark.asyncio
async def test_rejected_base64_as_422_logs_the_setting_to_change(monkeypatch):
    """A server that validates the format as an enum answers 422; the hint still fits."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="openai/m",
            dimensions=2,
            api_key="test-key",
            endpoint="http://localhost:8000/v1",
            encoding_format="base64",
        )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    error = mod.litellm.exceptions.UnprocessableEntityError(
        message="encoding_format: Input should be 'float'",
        model="openai/m",
        llm_provider="openai",
        response=httpx.Response(422, request=httpx.Request("POST", "http://localhost:8000/v1")),
    )
    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(side_effect=error))
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(mod.litellm.exceptions.UnprocessableEntityError):
        await engine.embed_text(["hello"])

    logged.assert_called_with(mod.ENCODING_FORMAT_HINT)


@pytest.mark.asyncio
async def test_default_base64_does_not_blame_an_unset_setting(monkeypatch):
    """On OpenAI's API base64 is the default, so a 400 there must not name the setting."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "api_base", None)
    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="openai/text-embedding-ada-002", dimensions=2, api_key="test-key"
        )
    assert engine.encoding_format == "base64"

    error = mod.litellm.exceptions.BadRequestError(
        message="This model does not support specifying dimensions.",
        model="openai/text-embedding-ada-002",
        llm_provider="openai",
    )
    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(side_effect=error))
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(mod.litellm.exceptions.BadRequestError):
        await engine.embed_text(["hello"])

    assert all(call.args[:1] != (mod.ENCODING_FORMAT_HINT,) for call in logged.call_args_list)


@pytest.mark.asyncio
async def test_client_side_refusal_of_the_format_gets_the_hint(monkeypatch):
    """litellm refuses encoding_format before sending for some providers (gemini, ollama);
    its message names the parameter, so the hint fits and the call stops at once."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            model="gemini/text-embedding-004",
            dimensions=2,
            api_key="test-key",
            encoding_format="base64",
        )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    error = mod.litellm.exceptions.UnsupportedParamsError(
        message="gemini does not support parameters: {'encoding_format': 'base64'}",
        model="gemini/text-embedding-004",
        llm_provider="gemini",
    )
    aembedding = AsyncMock(side_effect=error)
    monkeypatch.setattr(mod.litellm, "aembedding", aembedding)
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(mod.litellm.exceptions.UnsupportedParamsError):
        await engine.embed_text(["hello"])

    logged.assert_called_with(mod.ENCODING_FORMAT_HINT)
    assert aembedding.await_count == 1

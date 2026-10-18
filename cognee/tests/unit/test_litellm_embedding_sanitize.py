import base64
import re
import struct
import subprocess
import sys
import textwrap
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import litellm
import pytest
from tenacity import stop_after_attempt, wait_none

from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)


@pytest.fixture(autouse=True)
def _no_route_overrides(monkeypatch):
    """The tests choose the route; a developer's .env or litellm settings must not."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "USE_LITELLM_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "use_litellm_proxy", False)


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
async def test_a_base64_vector_with_nan_is_requested_again(monkeypatch):
    """A base64 reply can carry NaN as a transient glitch, which a retry clears."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider="openai", model="openai/text-embedding-3-large", dimensions=2, api_key="k"
        )
    assert engine.encoding_format == "base64"
    retry = engine.embed_text.retry_with(stop=stop_after_attempt(2), wait=wait_none())
    engine.embed_text = retry.__get__(engine)

    good = [0.5, 0.25]
    replies = [struct.pack("<2f", float("nan"), 0.25), struct.pack("<2f", *good)]
    aembedding = AsyncMock(
        side_effect=[
            SimpleNamespace(data=[{"embedding": base64.b64encode(reply).decode()}])
            for reply in replies
        ]
    )

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", aembedding)

    assert await engine.embed_text(["hello"]) == [good]
    assert aembedding.await_count == 2


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


_LINE_2_10 = (
    "Expected `list[float]` but got `str` with value `'AAAA'` - serialized value may not be "
    "as expected"
)
_LINE_2_11 = (
    "PydanticSerializationUnexpectedValue(Expected `list[float]` - serialized value may not "
    "be as expected [input_value='AAAA', input_type=str])"
)
_LINE_2_12 = (
    "PydanticSerializationUnexpectedValue(Expected `list[float]` - serialized value may not "
    "be as expected [field_name='embedding', input_value='AAAA', input_type=str])"
)
_UNRELATED_LINE = (
    "PydanticSerializationUnexpectedValue(Expected `int` - serialized value may not be as "
    "expected [field_name='prompt_tokens', input_value='7', input_type=str])"
)


def _warning(*lines):
    return "\n  ".join(["Pydantic serializer warnings:", *lines])


@pytest.mark.parametrize(
    ("message", "ignored"),
    [
        # One line per base64 vector, in each pydantic wording; long values are shortened.
        (_warning(_LINE_2_10), True),
        (_warning(_LINE_2_11), True),
        (_warning(_LINE_2_12, _LINE_2_12.replace("'AAAA'", "'7n6FPQsb...ozbXPg=='")), True),
        # Anything else in the same warning keeps all of it visible.
        (_warning(_LINE_2_12, _UNRELATED_LINE), False),
        # Another field, a value that is not base64, or another type is not litellm's reply.
        (_warning(_LINE_2_12.replace("'embedding'", "'values'")), False),
        (_warning(_LINE_2_12.replace("'AAAA'", "'0.1,0.2,0.3'")), False),
        (_warning(_LINE_2_12.replace("input_type=str", "input_type=dict")), False),
        (_warning(_LINE_2_12.replace("list[float]", "int")), False),
    ],
)
def test_only_a_warning_about_base64_vectors_is_ignored(message, ignored):
    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    # The warnings module matches the start of the message, ignoring case, like this.
    pattern = re.compile(mod._BASE64_SERIALIZER_WARNING, re.IGNORECASE)
    assert bool(pattern.match(message)) is ignored


def _dump_base64_reply(**usage):
    """What litellm does with a base64 reply: dump the openai SDK model that holds it."""
    from openai.types import CreateEmbeddingResponse, Embedding
    from openai.types.create_embedding_response import Usage

    # Long enough for pydantic to shorten the value in its warning.
    encoded = base64.b64encode(struct.pack("<768f", *([0.25] * 768))).decode()
    data = [
        Embedding.model_construct(embedding=encoded, index=i, object="embedding") for i in range(3)
    ]
    usage = Usage.model_construct(**{"prompt_tokens": 1, "total_tokens": 1, **usage})
    CreateEmbeddingResponse.model_construct(
        data=data, model="m", object="list", usage=usage
    ).model_dump()


def test_the_real_base64_warning_is_ignored_and_other_problems_are_not():
    from pydantic import VERSION as PYDANTIC_VERSION
    from pydantic import BaseModel

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    class Scores(BaseModel):
        values: list[float]

    class Record(BaseModel):
        vector: list[float]
        count: int

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _dump_base64_reply()
        assert len(caught) == 1, "pydantic no longer warns here; the filter can go"
        caught.clear()

        mod._ignore_base64_serializer_warning()
        _dump_base64_reply()
        assert caught == [], "pydantic reworded the warning; update the pattern"

        # Another problem in the same dump, another model's list[float] holding a str,
        # and a str that is in another field: all still shown.
        _dump_base64_reply(prompt_tokens="7")
        Scores.model_construct(values="0.1,0.2,0.3").model_dump()
        Record.model_construct(vector={"x": 1}, count="three").model_dump()
        expected = 3
        if tuple(int(part) for part in PYDANTIC_VERSION.split(".")[:2]) >= (2, 12):
            # A base64-looking string in another field. Before 2.12 no field is named.
            Scores.model_construct(values="AAAA").model_dump()
            expected += 1
        assert len(caught) == expected


@pytest.mark.skipif(sys.version_info < (3, 14), reason="context-aware warnings came in 3.14")
def test_the_filter_is_put_first_in_the_filters_in_effect():
    """With context-aware warnings, catch_warnings() keeps its filters in a context
    variable while warnings.filters stays the outer list. The filter already being first
    in the outer list must not stop it being put first inside the block."""
    import inspect

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    # The flag is fixed at startup, so this runs in a child process: the function's own
    # source with its pattern, and the dump litellm does, without importing cognee.
    script = "\n".join(
        [
            "import warnings",
            f"_BASE64_SERIALIZER_WARNING = {mod._BASE64_SERIALIZER_WARNING!r}",
            inspect.getsource(mod._ignore_base64_serializer_warning),
            textwrap.dedent(
                """
                import base64, struct
                from openai.types import CreateEmbeddingResponse, Embedding
                from openai.types.create_embedding_response import Usage

                encoded = base64.b64encode(struct.pack("<2f", 0.5, 0.25)).decode()
                data = [Embedding.model_construct(embedding=encoded, index=0, object="embedding")]
                reply = CreateEmbeddingResponse.model_construct(
                    data=data,
                    model="m",
                    object="list",
                    usage=Usage.model_construct(prompt_tokens=1, total_tokens=1),
                )
                _ignore_base64_serializer_warning()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "error", message="Pydantic serializer warnings", category=UserWarning
                    )
                    _ignore_base64_serializer_warning()
                    reply.model_dump()
                print("dumped without error")
                """
            ),
        ]
    )
    result = subprocess.run(
        [sys.executable, "-X", "context_aware_warnings=1", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "dumped without error" in result.stdout.splitlines(), result.stderr[-3000:]


@pytest.mark.asyncio
async def test_a_base64_reply_survives_a_strict_warning_filter(monkeypatch):
    """pytest drops a filter added at import once that test ends, and an "error" filter
    added after it would turn litellm's warning into a failed embedding call."""
    monkeypatch.setenv("MOCK_EMBEDDING", "false")

    with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
        engine = LiteLLMEmbeddingEngine(
            provider="openai", model="openai/text-embedding-3-large", dimensions=4, api_key="k"
        )
    assert engine.encoding_format == "base64"
    # A failure would otherwise run the whole retry ladder before the test reports it.
    engine.embed_text = engine.embed_text.retry_with(stop=stop_after_attempt(1)).__get__(engine)

    vector = [0.5, -0.25, 0.125, 1.0]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()

    async def fake_aembedding(**kwargs):
        _dump_base64_reply()  # litellm dumps the reply, which warns
        return SimpleNamespace(data=[{"embedding": encoded}])

    import cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine as mod

    monkeypatch.setattr(mod.litellm, "aembedding", fake_aembedding)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Pydantic serializer warnings", category=UserWarning
        )
        result = await engine.embed_text(["hello"])

    assert result == [vector]


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
@pytest.mark.parametrize("status", [400, 422])
async def test_default_base64_does_not_blame_an_unset_setting(monkeypatch, status):
    """On OpenAI's API base64 is the default, so even a refusal of it must not name the
    setting: the user never set it. Both status branches carry the hint."""
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

    if status == 400:
        error = mod.litellm.exceptions.BadRequestError(
            message="unsupported encoding_format",
            model="openai/text-embedding-ada-002",
            llm_provider="openai",
        )
    else:
        error = mod.litellm.exceptions.UnprocessableEntityError(
            message="encoding_format: Input should be 'float'",
            model="openai/text-embedding-ada-002",
            llm_provider="openai",
            response=httpx.Response(422, request=httpx.Request("POST", "https://api.openai.com")),
        )
    monkeypatch.setattr(mod.litellm, "aembedding", AsyncMock(side_effect=error))
    logged = Mock()
    monkeypatch.setattr(mod.logger, "error", logged)

    with pytest.raises(type(error)):
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

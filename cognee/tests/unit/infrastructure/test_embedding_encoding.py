"""embeddings/encoding.py: which routes request base64, and how responses decode."""

import base64
import importlib
import struct
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import litellm
import openai
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError, EmbeddingException
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig
from cognee.infrastructure.databases.vector.embeddings.encoding import (
    base64_supported,
    decode_embedding,
    rejects_encoding_format,
    resolve_encoding_format,
)
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    create_embedding_engine,
)
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)

# The module, not the function the package re-exports under the same name.
_FACTORY = importlib.import_module(
    "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine"
)


@pytest.fixture(autouse=True)
def _no_route_overrides(monkeypatch):
    """The tests choose the route; a developer's .env or litellm settings must not."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "USE_LITELLM_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "use_litellm_proxy", False)


@pytest.mark.parametrize(
    ("model", "endpoint", "expected"),
    [
        # OpenAI's own API: base64.
        ("openai/text-embedding-3-large", None, "base64"),
        ("text-embedding-3-small", None, "base64"),
        ("text-embedding-ada-002", None, "base64"),
        ("openai/text-embedding-3-large", "https://api.openai.com/v1", "base64"),
        # Everything else keeps float.
        ("openai/text-embedding-3-large", "https://proxy.invalid/v1", "float"),
        ("openai/BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("azure/my-deployment", "https://example.openai.azure.com", "float"),
        ("openrouter/openai/text-embedding-3-small", None, "float"),
        ("gemini/text-embedding-004", None, "float"),
        ("bedrock/amazon.titan-embed-text-v2:0", None, "float"),
        ("nvidia_nim/nv-embedqa-e5-v5", None, "float"),
        ("hosted_vllm/BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("BAAI/bge-m3", "http://localhost:8000/v1", "float"),
        ("nomic-embed-text", None, "float"),
        (None, None, "float"),
    ],
)
def test_default_format_per_route(model, endpoint, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    assert resolve_encoding_format(model, endpoint, None) == expected


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        # litellm sends an endpoint-less OpenAI request to these when set.
        ("OPENAI_BASE_URL", "http://localhost:4000", "float"),
        ("OPENAI_API_BASE", "https://gateway.example.com/v1", "float"),
        ("litellm.api_base", "http://localhost:4000", "float"),
        ("OPENAI_BASE_URL", "https://api.openai.com/v1", "base64"),
    ],
)
def test_default_format_follows_litellm_base_url(setting, value, expected, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    if setting == "litellm.api_base":
        monkeypatch.setattr(litellm, "api_base", value)
    else:
        monkeypatch.setenv(setting, value)

    assert resolve_encoding_format("openai/text-embedding-3-large", None, None) == expected
    # An explicit endpoint still wins over litellm's fallbacks.
    assert (
        resolve_encoding_format("openai/text-embedding-3-large", "https://api.openai.com/v1", None)
        == "base64"
    )


@pytest.mark.parametrize("switch", ["USE_LITELLM_PROXY", "litellm.use_litellm_proxy"])
def test_default_format_stays_float_when_litellm_routes_to_its_proxy(switch, monkeypatch):
    """litellm sends even an openai/ model to a LiteLLM proxy when told to, and the
    proxy's backend may refuse base64, so only an explicit setting asks for it there."""
    if switch == "USE_LITELLM_PROXY":
        monkeypatch.setenv("USE_LITELLM_PROXY", "true")
    else:
        monkeypatch.setattr(litellm, "use_litellm_proxy", True)

    assert resolve_encoding_format("openai/text-embedding-3-large", None, None) == "float"
    assert resolve_encoding_format("text-embedding-3-large", None, None) == "float"
    assert resolve_encoding_format("openai/text-embedding-3-large", None, "base64") == "base64"


def test_a_prefix_litellm_cannot_route_is_not_asked_about(capsys):
    """litellm's provider prefixes are case-sensitive, and asking it about one it does not
    know prints its provider list to stdout."""
    assert resolve_encoding_format("OpenAI/text-embedding-3-large", None, None) == "float"
    assert "Provider List" not in capsys.readouterr().out


@pytest.mark.parametrize("setting", ["EMBEDDING_ENDPOINT", "OPENAI_BASE_URL"])
def test_an_endpoint_that_does_not_parse_is_not_openais_api(setting, monkeypatch):
    """urlparse raises on an unclosed IPv6 bracket. The engine reports a bad
    EMBEDDING_ENDPOINT itself, so picking the format must not fail first."""
    endpoint = "http://[fd00::1:8080/v1"
    if setting == "OPENAI_BASE_URL":
        monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
        endpoint = None
    assert resolve_encoding_format("openai/text-embedding-3-large", endpoint, None) == "float"


@pytest.mark.parametrize("configured", ["base64", "float"])
def test_configured_format_wins(configured):
    assert resolve_encoding_format("openai/text-embedding-3-large", None, configured) == configured
    assert resolve_encoding_format("openai/m", "http://localhost:8000/v1", configured) == configured


def test_decode_embedding_handles_both_formats():
    vector = [0.0123, -0.5, 0.25, 1e-7]
    float32_vector = [struct.unpack("<f", struct.pack("<f", v))[0] for v in vector]
    encoded = base64.b64encode(struct.pack("<4f", *vector)).decode()

    decoded = decode_embedding(encoded, "base64")

    assert decoded == float32_vector
    assert all(isinstance(v, float) for v in decoded)
    assert decode_embedding(float32_vector, "float") is float32_vector


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        # A LiteLLM proxy in front of Bedrock answers base64 with 1-bit vectors as a list.
        ([1.0, 0.0, 1.0, 1.0], "got list"),
        ("not base64!", "does not decode"),
        # Five bytes: not a whole number of float32 values.
        (base64.b64encode(b"\x00\x00\x00\x00\x00").decode(), "does not decode"),
        # Characters outside the base64 alphabet are refused, not skipped.
        ("AAAA AAAA AAAA AAAA", "does not decode"),
    ],
)
def test_decode_embedding_rejects_what_is_not_base64_float32(payload, detail):
    with pytest.raises(EmbeddingEncodingError, match=detail) as info:
        decode_embedding(payload, "base64")
    assert "EMBEDDING_ENCODING_FORMAT" in str(info.value)


@pytest.mark.parametrize(
    ("provider", "supported"),
    [
        # Engines that send an OpenAI-style request: trusted, the reply is checked.
        ("openai", True),
        ("custom", True),
        ("openai_compatible", True),
        ("azure", True),
        ("bedrock", True),
        # Engines with no wire format to request.
        ("fastembed", False),
        ("ollama", False),
    ],
)
def test_base64_supported_everywhere_but_engines_without_a_wire(provider, supported):
    assert base64_supported(provider) is supported


def _create(provider, model, encoding_format):
    return create_embedding_engine(
        provider, model, 4, None, None, None, None, 36, None, None, None, None, encoding_format
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [("fastembed", "BAAI/bge-small-en-v1.5"), ("ollama", "nomic-embed-text")],
)
def test_base64_on_an_engine_without_a_wire_is_a_configuration_error(provider, model):
    with pytest.raises(CogneeConfigurationError, match="EMBEDDING_ENCODING_FORMAT"):
        _create(provider, model, "base64")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # A proxy or gateway route: cognee cannot see what is behind it.
        ("custom", "litellm_proxy/text-embedding-3-large"),
        ("custom", "openai/text-embedding-3-large"),
        # litellm turns base64 into Bedrock's own type (Titan v2: 1-bit vectors, as a
        # list), and forwards it to Azure unchanged. The first reply shows which held.
        ("bedrock", "bedrock/amazon.titan-embed-text-v2:0"),
        ("azure", "azure/my-deployment"),
    ],
)
def test_configured_base64_is_trusted_at_creation(provider, model, monkeypatch):
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    engine = _create(provider, model, "base64")
    assert engine.encoding_format == "base64"


@pytest.mark.parametrize("value", ["Base64", "FLOAT", "b64", ""])
def test_a_value_outside_the_two_formats_is_a_configuration_error(value, monkeypatch):
    """The env path is typed; the SDK setter is not, so the factory checks the value,
    also for the engines that have no wire format to apply it to."""
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    for provider, model in [
        ("openai", "openai/text-embedding-3-large"),
        ("openai_compatible", "nomic-embed-text"),
        ("fastembed", "BAAI/bge-small-en-v1.5"),
        ("ollama", "nomic-embed-text"),
    ]:
        with pytest.raises(CogneeConfigurationError, match="must be 'base64' or 'float'"):
            _create(provider, model, value)


@pytest.mark.parametrize(
    "payload",
    [
        # int8 values [0, 0, -128, 127] read as float32 are +infinity.
        base64.b64encode(bytes([0, 0, 0x80, 0x7F])).decode(),
        base64.b64encode(struct.pack("<2f", 0.5, float("nan"))).decode(),
    ],
)
def test_nan_or_infinity_is_retried_rather_than_blamed_on_base64(payload):
    """A base64 reply can carry NaN as a transient glitch, which a retry clears, so it
    is not the terminal error that tells the user base64 is unsupported."""
    with pytest.raises(EmbeddingException, match="NaN or infinity") as info:
        decode_embedding(payload, "base64")
    assert not isinstance(info.value, EmbeddingEncodingError)


def test_decode_embedding_rejects_an_empty_payload():
    with pytest.raises(EmbeddingEncodingError, match="empty"):
        decode_embedding("", "base64")


async def _sdk_error(status, body):
    """The error the openai SDK raises for this reply, built by the SDK itself."""
    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
        ),
    )
    with pytest.raises(openai.APIStatusError) as info:
        await client.embeddings.create(model="m", input=["a"], encoding_format="base64")
    return info.value


async def _litellm_error(model, status, body):
    """The error litellm itself raises for this reply: through the openai SDK for an
    openai/ model, through its own HTTP handler for hosted_vllm/ and the like."""
    reply = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    if model.startswith("openai/"):
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://localhost:8000/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=reply),
        )
    else:
        # Set after construction: AsyncHTTPHandler takes a transport only from litellm 1.105.
        client = AsyncHTTPHandler()
        client.client = httpx.AsyncClient(transport=reply)
    with pytest.raises(openai.APIStatusError) as info:
        await litellm.aembedding(
            model=model,
            input=["a"],
            encoding_format="base64",
            api_base="http://localhost:8000/v1",
            api_key="test-key",
            client=client,
        )
    return info.value


_ECHOED_REQUEST = {"input": ["a"], "model": "m", "encoding_format": "base64"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "refused"),
    [
        # The server refuses the encoding, in the shapes servers say it.
        (
            422,
            {
                "detail": [
                    {
                        "type": "literal_error",
                        "loc": ["body", "encoding_format"],
                        "msg": "Input should be 'float'",
                        "input": "base64",
                    }
                ]
            },
            True,
        ),
        (400, {"error": {"message": "Invalid value", "param": "encoding_format"}}, True),
        (400, {"error": {"message": "Unsupported encoding_format: base64", "param": None}}, True),
        (
            422,
            {
                "error": "Failed to deserialize the JSON body into the target type: "
                "encoding_format: unknown variant `base64`, expected `float`",
                "error_type": "Unprocessable Entity",
            },
            True,
        ),
        # The openai SDK keeps only "error" ("Bad Request") of this reply.
        (
            400,
            {
                "statusCode": 400,
                "code": "FST_ERR_VALIDATION",
                "error": "Bad Request",
                "message": "body/encoding_format must be equal to one of the allowed values",
            },
            True,
        ),
        (
            400,
            {
                "error": {
                    "message": "Provider returned error",
                    "code": 400,
                    "metadata": {
                        "raw": '{"error": {"message": "Unsupported encoding_format: base64"}}',
                        "provider_name": "Example",
                    },
                }
            },
            True,
        ),
        # Errors keyed by field name.
        (400, {"encoding_format": ["is not included in the list"]}, True),
        # Errors about something else, echoing the request or the input texts.
        (
            422,
            {
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "input_type"],
                        "msg": "Field required",
                        "input": _ECHOED_REQUEST,
                    }
                ]
            },
            False,
        ),
        (
            422,
            {
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["body"],
                        "msg": "Value error, model 'm' is not served here",
                        "input": _ECHOED_REQUEST,
                    }
                ]
            },
            False,
        ),
        (
            422,
            {
                "detail": [
                    {
                        "type": "too_long",
                        "loc": ["body", "input"],
                        "msg": "List should have at most 1 item after validation, not 2",
                        "input": ["a", "Images are inlined as base64 data URIs."],
                    }
                ]
            },
            False,
        ),
        (400, {"error": {"message": "The model `m` does not exist", "param": None}}, False),
    ],
)
async def test_a_refusal_is_read_from_the_error_not_the_echoed_request(status, body, refused):
    assert rejects_encoding_format(await _sdk_error(status, body)) is refused
    for model in ("openai/m", "hosted_vllm/m", "openrouter/openai/text-embedding-3-small"):
        assert rejects_encoding_format(await _litellm_error(model, status, body)) is refused, model


@pytest.mark.parametrize(
    ("message", "refused"),
    [
        # litellm refuses the parameter before sending, so there is no server body.
        ("gemini does not support parameters: {'encoding_format': 'base64'}", True),
        ("This model does not support specifying dimensions.", False),
    ],
)
def test_an_error_without_a_body_is_matched_on_its_message(message, refused):
    error = litellm.exceptions.UnsupportedParamsError(
        message=message, model="gemini/text-embedding-004", llm_provider="gemini"
    )
    assert rejects_encoding_format(error) is refused


def _llm(api_key):
    return SimpleNamespace(
        llm_provider="openai", llm_api_key=api_key, llm_azure_use_managed_identity=False
    )


def _engine_from_env(monkeypatch, llm_api_key, **env):
    """get_embedding_engine() for these environment variables, read by EmbeddingConfig."""
    for var in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_API_KEY",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_API_BASE",
        "EMBEDDING_ENCODING_FORMAT",
        "COGNEE_SKIP_PREFLIGHT",
        "COGNEE_SKIP_CONNECTION_TEST",
        "MOCK_EMBEDDING",
    ):
        monkeypatch.delenv(var, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = EmbeddingConfig(_env_file=None)
    monkeypatch.setattr(_FACTORY, "get_embedding_context_config", lambda: config)
    monkeypatch.setattr(_FACTORY, "get_llm_context_config", lambda: _llm(llm_api_key))
    create_embedding_engine.cache_clear()
    try:
        with patch.object(LiteLLMEmbeddingEngine, "get_tokenizer", return_value=Mock()):
            return _FACTORY.get_embedding_engine()
    finally:
        create_embedding_engine.cache_clear()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # OpenAI's own API defaults to base64, and the setting opts out.
        ({"EMBEDDING_ENCODING_FORMAT": "float"}, "float"),
        # A proxy defaults to float, and the setting opts in.
        (
            {
                "EMBEDDING_ENDPOINT": "http://localhost:4000/v1",
                "EMBEDDING_ENCODING_FORMAT": "base64",
            },
            "base64",
        ),
    ],
)
def test_the_setting_reaches_the_engine_from_the_environment(env, expected, monkeypatch):
    assert _engine_from_env(monkeypatch, "test-key", **env).encoding_format == expected


def test_a_keyless_run_leaves_base64_for_the_provider_a_key_selects(monkeypatch):
    """With no key, embeddings run on local fastembed, which has no wire format. The
    setting is not applied, and that is said once, instead of failing a run that works
    without it."""
    import cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine as local
    from cognee.infrastructure.databases.vector.embeddings import config as config_module

    class _LocalEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(config_module, "_resolve_embedding_dimensions", lambda *args: 384)
    monkeypatch.setattr(local, "FastembedEmbeddingEngine", _LocalEngine)
    warned = Mock()
    monkeypatch.setattr(_FACTORY.logger, "warning", warned)
    _FACTORY._warn_base64_waits_for_a_key.cache_clear()
    try:
        first = _engine_from_env(monkeypatch, None, EMBEDDING_ENCODING_FORMAT="base64")
        second = _engine_from_env(monkeypatch, None, EMBEDDING_ENCODING_FORMAT="base64")
    finally:
        _FACTORY._warn_base64_waits_for_a_key.cache_clear()

    assert isinstance(first, _LocalEngine)
    assert isinstance(second, _LocalEngine)
    warned.assert_called_once()
    assert "EMBEDDING_ENCODING_FORMAT=base64" in warned.call_args.args[0]

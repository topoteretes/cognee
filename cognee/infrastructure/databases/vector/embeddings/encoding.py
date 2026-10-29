"""Wire format of embedding responses: base64 or float JSON.

An OpenAI-style embeddings API returns each vector either as a JSON array of
decimals ("float") or as one base64 string of little-endian float32 values
("base64"). The values are the same; base64 is far cheaper to parse, because
the openai SDK builds a float response one element at a time.

base64 is used automatically only on routes known to accept it: OpenAI's own
API. Every other route sends the request cognee always sent and gets floats,
unless EMBEDDING_ENCODING_FORMAT=base64 opts it in. An opted-in route is
trusted and checked on the first reply: a provider that cannot honour base64
answers with a list, bad bytes or a 400/422, and each fails that call for good.
"""

import base64
import binascii
import os
import re
from typing import Literal
from urllib.parse import urlparse

import httpx
import litellm
import numpy as np
import openai

from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError, EmbeddingException

EncodingFormat = Literal["base64", "float"]
ENCODING_FORMATS = ("base64", "float")

ENCODING_FORMAT_HINT = (
    "EMBEDDING_ENCODING_FORMAT=base64 is set, and this route rejected it. "
    "Fix: comment out or remove EMBEDDING_ENCODING_FORMAT in your .env file or environment."
)


def validate_encoding_format(configured: str | None) -> EncodingFormat | None:
    """The configured format, or a configuration error for anything but the two values.

    The env path is typed, but cognee.config.set_embedding_config() stores whatever it
    is given, and a value like "Base64" would send no format and then reject the reply.
    """
    if configured is None or configured in ENCODING_FORMATS:
        return configured
    raise CogneeConfigurationError(
        f"EMBEDDING_ENCODING_FORMAT must be 'base64' or 'float', got {configured!r}.",
        remediation="Set EMBEDDING_ENCODING_FORMAT to base64 or float, or remove it.",
    )


_OPENAI_API_HOST = "api.openai.com"


def _routes_through_openai_handler(model: str | None) -> bool:
    """Whether litellm sends this model through its OpenAI handler, which forwards
    encoding_format unchanged. Other handlers translate or reject it."""
    if not model:
        return False
    if "/" in model:
        # litellm's provider prefixes are case-sensitive: it cannot route "OpenAI/...".
        return model.split("/", 1)[0] == "openai"
    # An unprefixed model resolves by litellm's model registry.
    return litellm.model_cost.get(model, {}).get("litellm_provider") == "openai"


def _litellm_openai_endpoint(endpoint: str | None) -> str | None:
    """The base URL litellm's OpenAI handler sends to, in litellm's own order.

    Mirrors litellm's lookup (api_base, litellm.api_base, OPENAI_BASE_URL,
    OPENAI_API_BASE), which cognee cannot call before the request is built. A
    mismatch only affects the default: base64 to a proxy that then rejects it, loudly.
    """
    return (
        endpoint or litellm.api_base or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )


def _hostname(url: str) -> str | None:
    """The URL's host, or None when the URL does not parse (an unclosed IPv6 bracket)."""
    try:
        return urlparse(url).hostname
    except ValueError:
        return None


def _routes_to_openai_api(model: str | None, endpoint: str | None) -> bool:
    """Whether litellm sends this model to OpenAI's own embeddings API."""
    endpoint = _litellm_openai_endpoint(endpoint)
    if endpoint and _hostname(endpoint) != _OPENAI_API_HOST:
        return False
    if model is None or not _routes_through_openai_handler(model):
        return False
    # litellm has the last word: USE_LITELLM_PROXY (or litellm.use_litellm_proxy) sends
    # even an openai/ model to a LiteLLM proxy, whose backend may refuse base64. Asked
    # only about models that are OpenAI's by name, so it never meets one it does not know.
    # It reports every failure as BadRequestError.
    try:
        return litellm.get_llm_provider(model=model, api_base=endpoint)[1] == "openai"
    except litellm.exceptions.BadRequestError:
        return False


def base64_supported(provider: str) -> bool:
    """Whether EMBEDDING_ENCODING_FORMAT=base64 can be sent at all for this provider.

    Only the engines with no wire format are refused: fastembed embeds in-process and
    the ollama engine calls Ollama's native API, which has no such parameter. Every
    other engine sends the request and decode_embedding() checks the reply.
    """
    return provider not in ("fastembed", "ollama")


def resolve_encoding_format(
    model: str | None, endpoint: str | None, configured: EncodingFormat | None
) -> EncodingFormat:
    """The format to request: the configured one, else base64 only for OpenAI's own API."""
    configured = validate_encoding_format(configured)
    if configured is not None:
        return configured
    return "base64" if _routes_to_openai_api(model, endpoint) else "float"


def decode_embedding(embedding: list[float] | str, encoding_format: EncodingFormat) -> list[float]:
    """Return the vector as floats.

    With base64 requested, the server must answer with a base64 string of little-endian
    float32 values. A list means a provider or proxy translated the request into its own
    format (a LiteLLM proxy in front of Bedrock returns 1-bit vectors as a list of 0/1),
    so it raises EmbeddingEncodingError instead of storing garbage. With float, the list
    passes through unchanged.

    The length is not checked against the configured dimensions: with EMBEDDING_DIMENSIONS
    unset, the connection test adopts the length of the first vector it gets.
    """
    if encoding_format == "float":
        return embedding
    if not isinstance(embedding, str):
        raise EmbeddingEncodingError(f"got {type(embedding).__name__}")
    try:
        vector = np.frombuffer(base64.b64decode(embedding, validate=True), dtype="<f4")
    except (binascii.Error, ValueError) as error:
        raise EmbeddingEncodingError(f"payload does not decode as float32: {error}") from error
    if not vector.size:
        raise EmbeddingEncodingError("payload is empty")
    # NaN or infinity cannot be stored. A base64 reply can carry them as a transient glitch,
    # which a retry clears, or as bytes of another type read as float32: int8, uint8 and
    # byte-swapped float32 usually decode to them, float16 and float64 never do. So this is
    # a retried EmbeddingException, not the terminal error that blames base64 support.
    if not np.isfinite(vector).all():
        raise EmbeddingException(
            "The embedding server returned NaN or infinity in a base64 vector. If this "
            "persists, the server may not send float32: set EMBEDDING_ENCODING_FORMAT=float."
        )
    return vector.tolist()


_ENCODING_REJECTION_RE = re.compile(r"encoding_format|base64", re.IGNORECASE)
# Where an error body repeats the request back: pydantic's "input" holds the rejected
# value or the whole request, and "ctx" the constraint. On a base64 call those name
# encoding_format and base64 whatever the error is about, so they are not read.
_ECHO_KEYS = frozenset({"input", "ctx", "request"})


def _server_error_body(error: BaseException) -> object | None:
    """The JSON body of the HTTP reply behind ``error``, or None.

    Read from the reply itself, which the openai SDK's errors and httpx's carry. litellm
    raises its own error with a placeholder reply and keeps one of those in the chain,
    linked through the implicit context. The SDK's ``body`` is a last resort: it keeps
    only the reply's "error" member.
    """
    current: BaseException | None = error
    for _ in range(6):
        if current is None:
            return None
        if isinstance(current, (openai.APIStatusError, httpx.HTTPStatusError)):
            response = getattr(current, "response", None)
            if isinstance(response, httpx.Response):
                try:
                    return response.json()
                except (ValueError, httpx.StreamError):
                    pass
            # Not every subclass sets it: litellm's UnsupportedParamsError has no body.
            body = getattr(current, "body", None)
            if body is not None:
                return body
        current = current.__cause__ or current.__context__
    return None


def _names_encoding(body: object) -> bool:
    """Whether an error body is about the encoding, read everywhere but its echoes."""
    if isinstance(body, str):
        return bool(_ENCODING_REJECTION_RE.search(body))
    if isinstance(body, list):
        return any(_names_encoding(item) for item in body)
    if isinstance(body, dict):
        # A key can be the field itself, in errors keyed by field name.
        return any(
            key == "encoding_format" or _names_encoding(value)
            for key, value in body.items()
            if key not in _ECHO_KEYS
        )
    return False


def rejects_encoding_format(error: BaseException) -> bool:
    """Whether a 400/422 is about the encoding, so ENCODING_FORMAT_HINT fits it.

    A server that rejects base64 says so ("unsupported encoding_format", a validation
    error located at encoding_format); a bad model name or dimensions does not, and the
    hint would send the user after the wrong fix. The server's JSON reply is read whole,
    except where it echoes the request, which names encoding_format and base64 on every
    call, or the input texts. An error with no reply, such as litellm refusing the
    parameter before sending, is matched on its message. So is a relay that copies the
    backend's error into its own message, as a LiteLLM proxy does, echo included.
    """
    body = _server_error_body(error)
    if body is not None:
        return _names_encoding(body)
    return bool(_ENCODING_REJECTION_RE.search(str(error)))

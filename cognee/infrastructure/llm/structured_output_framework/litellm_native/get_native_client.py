"""Factory for the litellm_native structured output client.

This module mirrors ``litellm_instructor/llm/get_llm_client.py`` but produces a
``NativeLiteLLMAdapter`` instead of a per-provider instructor adapter.  Because
the native adapter is universal (one class for all providers), the factory is
significantly simpler — no provider enum dispatch, just config → adapter.

Caching strategy
~~~~~~~~~~~~~~~~
We reuse the same ``lru_cache`` + frozen-dataclass-key pattern as the instructor
factory so that repeated calls with identical config return the same adapter
instance.  The cache key includes every config field that could change the
adapter's behaviour (model, endpoint, api key hash, etc.) and caps size via
``maxsize`` to avoid unbounded memory growth in long-running processes.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Hashable

from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.llm.structured_output_framework.litellm_native.llm_native_interface import (
    LLMNativeInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)

_NATIVE_CLIENT_CACHE_MAXSIZE = 32


# ---------------------------------------------------------------------------
# Cache key helpers — copied from get_llm_client.py to stay decoupled.
# ---------------------------------------------------------------------------

class _SecretCacheKey:
    """Cache key segment that compares secrets without exposing them in repr."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SecretCacheKey) and self.__value == other.__value

    def __hash__(self) -> int:
        return hash(self.__value)

    def __repr__(self) -> str:
        return "<redacted>" if self.__value else "<empty>"


_FROZEN_DICT = "__cognee_dict__"


def _freeze_for_cache(value: Any) -> Hashable:
    """Convert nested JSON-like config values into a deterministic hashable form."""
    if isinstance(value, dict):
        return (
            _FROZEN_DICT,
            tuple(sorted((str(k), _freeze_for_cache(v)) for k, v in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass(frozen=True)
class _NativeClientCacheKey:
    """Hashable representation of config fields that affect adapter creation."""

    model: str
    api_key_hash: _SecretCacheKey
    endpoint: str
    api_version: str | None
    streaming: bool
    max_completion_tokens: int
    fallback_api_key_hash: _SecretCacheKey
    fallback_endpoint: str
    fallback_model: str
    llm_args: Hashable


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=_NATIVE_CLIENT_CACHE_MAXSIZE)
def _get_native_client_cached(cache_key: _NativeClientCacheKey) -> NativeLiteLLMAdapter:
    """Create and cache a ``NativeLiteLLMAdapter`` keyed by config fields.

    The actual secret values (api_key, fallback_api_key) are read from the
    context config, not from the cache key (which only stores a hash).
    """
    llm_config = get_llm_context_config()

    return NativeLiteLLMAdapter(
        api_key=llm_config.llm_api_key or "",
        model=cache_key.model,
        max_completion_tokens=cache_key.max_completion_tokens,
        endpoint=cache_key.endpoint or None,
        api_version=cache_key.api_version,
        streaming=cache_key.streaming,
        fallback_model=cache_key.fallback_model or None,
        fallback_api_key=llm_config.fallback_api_key or None,
        fallback_endpoint=cache_key.fallback_endpoint or None,
        llm_args=dict(cache_key.llm_args) if isinstance(cache_key.llm_args, tuple) else None,
    )


def get_native_client(raise_api_key_error: bool = True) -> LLMNativeInterface:
    """Build (or retrieve from cache) a ``NativeLiteLLMAdapter``.

    This function is the public entry point called by ``LLMGateway`` when
    ``STRUCTURED_OUTPUT_FRAMEWORK="litellm_native"``.

    Args:
        raise_api_key_error: When ``True`` (default), raise
            ``LLMAPIKeyNotSetError`` if no API key is configured.  Set to
            ``False`` in contexts where the key may legitimately be absent
            (e.g. computing chunk sizes).

    Returns:
        A cached ``NativeLiteLLMAdapter`` instance.

    Raises:
        LLMAPIKeyNotSetError: If the API key is missing and
            *raise_api_key_error* is ``True``.
    """
    llm_config = get_llm_context_config()

    # Validate API key presence.
    api_key = llm_config.llm_api_key
    if raise_api_key_error and (api_key is None or api_key.strip() == ""):
        raise LLMAPIKeyNotSetError()

    # Compute effective max_completion_tokens — same logic as get_llm_client.
    from cognee.infrastructure.llm.utils import get_model_max_completion_tokens

    model_max = get_model_max_completion_tokens(llm_config.llm_model)
    user_max = llm_config.llm_max_completion_tokens
    if model_max is not None:
        max_completion_tokens = min(model_max, user_max)
    else:
        max_completion_tokens = user_max

    cache_key = _NativeClientCacheKey(
        model=llm_config.llm_model,
        api_key_hash=_SecretCacheKey(api_key or ""),
        endpoint=llm_config.llm_endpoint,
        api_version=llm_config.llm_api_version,
        streaming=llm_config.llm_streaming,
        max_completion_tokens=max_completion_tokens,
        fallback_api_key_hash=_SecretCacheKey(llm_config.fallback_api_key or ""),
        fallback_endpoint=llm_config.fallback_endpoint,
        fallback_model=llm_config.fallback_model,
        llm_args=_freeze_for_cache(llm_config.llm_args or {}),
    )

    return _get_native_client_cached(cache_key)

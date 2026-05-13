"""Get the LLM client."""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Hashable, TypeGuard

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.exceptions import (
    LLMAPIKeyNotSetError,
    UnsupportedLLMProviderError,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)

_LLM_CLIENT_CACHE_MAXSIZE = 32
_FROZEN_DICT = "__cognee_dict__"
_FROZEN_LIST = "__cognee_list__"
_FROZEN_TUPLE = "__cognee_tuple__"
_FROZEN_SET = "__cognee_set__"


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


@dataclass(frozen=True)
class _LLMClientCacheKey:
    """Hashable representation of LLM config fields that affect adapter creation."""

    provider: str
    model: str
    api_key_cache_key: _SecretCacheKey
    endpoint: str
    api_version: str | None
    instructor_mode: str
    streaming: bool
    max_completion_tokens: int
    transcription_model: str
    fallback_api_key_cache_key: _SecretCacheKey
    fallback_endpoint: str
    fallback_model: str
    llm_args: Hashable
    azure_use_managed_identity: bool
    llama_cpp_model_path: str | None
    llama_cpp_n_ctx: int
    llama_cpp_n_gpu_layers: int
    llama_cpp_chat_format: str


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    """
    Define an Enum for identifying different LLM Providers.

    This Enum includes the following members:
    - OPENAI: Represents the OpenAI provider.
    - OLLAMA: Represents the Ollama provider.
    - ANTHROPIC: Represents the Anthropic provider.
    - CUSTOM: Represents a custom provider option.
    - GEMINI: Represents the Gemini provider.
    - MISTRAL: Represents the Mistral AI provider.
    - BEDROCK: Represents the AWS Bedrock provider.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    AZURE = "azure"
    BEDROCK = "bedrock"
    LLAMA_CPP = "llama_cpp"


_API_KEY_REQUIRED_PROVIDERS = {
    LLMProvider.OPENAI,
    LLMProvider.OLLAMA,
    LLMProvider.CUSTOM,
    LLMProvider.GEMINI,
    LLMProvider.MISTRAL,
    LLMProvider.ANTHROPIC,
}


def _freeze_for_cache(value: Any) -> Hashable:
    """Convert nested JSON-like config values into a deterministic hashable form."""
    if isinstance(value, dict):
        return (
            _FROZEN_DICT,
            tuple(sorted((str(key), _freeze_for_cache(item)) for key, item in value.items())),
        )
    if isinstance(value, list):
        return (_FROZEN_LIST, tuple(_freeze_for_cache(item) for item in value))
    if isinstance(value, tuple):
        return (_FROZEN_TUPLE, tuple(_freeze_for_cache(item) for item in value))
    if isinstance(value, set):
        return (_FROZEN_SET, tuple(sorted((_freeze_for_cache(item) for item in value), key=repr)))

    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _is_frozen_mapping_payload(value: Any) -> TypeGuard[tuple[tuple[str, Hashable], ...]]:
    """Return whether a frozen payload represents sorted key-value mapping items."""
    if not isinstance(value, tuple):
        return False
    return all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value
    )


def _is_frozen_sequence_payload(value: Any) -> TypeGuard[tuple[Hashable, ...]]:
    """Return whether a frozen payload represents sequence items."""
    return isinstance(value, tuple)


def _unfreeze_from_cache(value: Any) -> Any:
    """Rebuild JSON-like values from their cache-key representation."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        kind, payload = value
        if kind == _FROZEN_DICT and _is_frozen_mapping_payload(payload):
            return {key: _unfreeze_from_cache(item) for key, item in payload}
        if kind == _FROZEN_LIST and _is_frozen_sequence_payload(payload):
            return [_unfreeze_from_cache(item) for item in payload]
        if kind == _FROZEN_TUPLE and _is_frozen_sequence_payload(payload):
            return tuple(_unfreeze_from_cache(item) for item in payload)
        if kind == _FROZEN_SET and _is_frozen_sequence_payload(payload):
            return {_unfreeze_from_cache(item) for item in payload}
    return value


def _secret_cache_key(secret: str | None) -> _SecretCacheKey:
    """Return a cache key segment that keeps secrets out of rendered cache keys."""
    return _SecretCacheKey(secret or "")


def _build_llm_client_cache_key(llm_config, max_completion_tokens: int) -> _LLMClientCacheKey:
    """Build a complete cache key for fields that affect LLM adapter construction."""
    return _LLMClientCacheKey(
        provider=llm_config.llm_provider,
        model=llm_config.llm_model,
        api_key_cache_key=_secret_cache_key(llm_config.llm_api_key),
        endpoint=llm_config.llm_endpoint,
        api_version=llm_config.llm_api_version,
        instructor_mode=llm_config.llm_instructor_mode.lower(),
        streaming=llm_config.llm_streaming,
        max_completion_tokens=max_completion_tokens,
        transcription_model=llm_config.transcription_model,
        fallback_api_key_cache_key=_secret_cache_key(llm_config.fallback_api_key),
        fallback_endpoint=llm_config.fallback_endpoint,
        fallback_model=llm_config.fallback_model,
        llm_args=_freeze_for_cache(llm_config.llm_args or {}),
        azure_use_managed_identity=llm_config.llm_azure_use_managed_identity,
        llama_cpp_model_path=llm_config.llama_cpp_model_path,
        llama_cpp_n_ctx=llm_config.llama_cpp_n_ctx,
        llama_cpp_n_gpu_layers=llm_config.llama_cpp_n_gpu_layers,
        llama_cpp_chat_format=llm_config.llama_cpp_chat_format,
    )


def _raise_for_missing_api_key(
    provider: LLMProvider,
    api_key: str | None,
    raise_api_key_error: bool,
    use_managed_identity: bool = False,
) -> None:
    """Preserve provider-specific API key validation before cache lookup."""
    requires_api_key = provider in _API_KEY_REQUIRED_PROVIDERS or (
        provider == LLMProvider.AZURE and not use_managed_identity
    )
    if requires_api_key and (api_key is None or api_key.strip() == "") and raise_api_key_error:
        raise LLMAPIKeyNotSetError()


@lru_cache(maxsize=_LLM_CLIENT_CACHE_MAXSIZE)
def _get_llm_client_cached(cache_key: _LLMClientCacheKey) -> LLMInterface:
    """Create and cache LLM adapters with bounded LRU eviction."""
    llm_config = get_llm_config()
    provider = LLMProvider(cache_key.provider)
    llm_api_key: str = llm_config.llm_api_key or ""
    llm_args = _unfreeze_from_cache(cache_key.llm_args) or {}
    max_completion_tokens = cache_key.max_completion_tokens

    if provider == LLMProvider.AZURE:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter import (
            AzureOpenAIAdapter,
        )

        return AzureOpenAIAdapter(
            api_key=llm_api_key,
            endpoint=cache_key.endpoint,
            api_version=cache_key.api_version,
            model=cache_key.model,
            transcription_model=cache_key.transcription_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=cache_key.instructor_mode,
            streaming=cache_key.streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=cache_key.fallback_endpoint,
            fallback_model=cache_key.fallback_model,
            llm_args=llm_args,
            use_managed_identity=cache_key.azure_use_managed_identity,
        )

    elif provider == LLMProvider.OPENAI:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
            OpenAIAdapter,
        )

        return OpenAIAdapter(
            api_key=llm_api_key,
            endpoint=cache_key.endpoint,
            api_version=cache_key.api_version,
            model=cache_key.model,
            transcription_model=cache_key.transcription_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=cache_key.instructor_mode,
            streaming=cache_key.streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=cache_key.fallback_endpoint,
            fallback_model=cache_key.fallback_model,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.OLLAMA:
        return OllamaAPIAdapter(
            cache_key.endpoint,
            llm_api_key,
            cache_key.model,
            "Ollama",
            max_completion_tokens,
            instructor_mode=cache_key.instructor_mode,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
            AnthropicAdapter,
        )

        return AnthropicAdapter(
            llm_api_key,
            cache_key.model,
            max_completion_tokens,
            instructor_mode=cache_key.instructor_mode,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.CUSTOM:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return GenericAPIAdapter(
            api_key=llm_api_key,
            model=cache_key.model,
            max_completion_tokens=max_completion_tokens,
            name="Custom",
            endpoint=cache_key.endpoint,
            instructor_mode=cache_key.instructor_mode,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=cache_key.fallback_endpoint,
            fallback_model=cache_key.fallback_model,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.GEMINI:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
            GeminiAdapter,
        )

        return GeminiAdapter(
            api_key=llm_api_key,
            model=cache_key.model,
            max_completion_tokens=max_completion_tokens,
            endpoint=cache_key.endpoint,
            api_version=cache_key.api_version,
            instructor_mode=cache_key.instructor_mode,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.MISTRAL:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
            MistralAdapter,
        )

        return MistralAdapter(
            api_key=llm_api_key,
            model=cache_key.model,
            max_completion_tokens=max_completion_tokens,
            endpoint=cache_key.endpoint,
            instructor_mode=cache_key.instructor_mode,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.BEDROCK:
        # if llm_config.llm_api_key is None and raise_api_key_error:
        #     raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
            BedrockAdapter,
        )

        return BedrockAdapter(
            model=cache_key.model,
            api_key=llm_config.llm_api_key,
            max_completion_tokens=max_completion_tokens,
            streaming=cache_key.streaming,
            instructor_mode=cache_key.instructor_mode,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.LLAMA_CPP:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter import (
            LlamaCppAPIAdapter,
        )

        return LlamaCppAPIAdapter(
            model=cache_key.model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=cache_key.instructor_mode,
            endpoint=cache_key.endpoint,
            api_key=llm_config.llm_api_key,
            model_path=cache_key.llama_cpp_model_path,
            n_ctx=cache_key.llama_cpp_n_ctx,
            n_gpu_layers=cache_key.llama_cpp_n_gpu_layers,
            chat_format=cache_key.llama_cpp_chat_format,
            llm_args=llm_args,
        )
    else:
        raise UnsupportedLLMProviderError(provider)


def get_llm_client(raise_api_key_error: bool = True) -> LLMInterface:
    """
    Get the LLM client based on the configuration using Enums.

    This function retrieves the configuration for the LLM provider and model, and
    initializes the appropriate LLM client adapter accordingly. It raises an
    LLMAPIKeyNotSetError if the LLM API key is not set for certain providers or if the provider
    is unsupported.

    Returns:
    --------

        An instance of the appropriate LLM client adapter based on the provider
        configuration.
    """
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)
    _raise_for_missing_api_key(
        provider,
        llm_config.llm_api_key,
        raise_api_key_error,
        llm_config.llm_azure_use_managed_identity,
    )

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_completion_tokens,
    )  # imported here to avoid circular imports

    model_max_completion_tokens = get_model_max_completion_tokens(llm_config.llm_model)
    user_max = llm_config.llm_max_completion_tokens
    if model_max_completion_tokens is not None:
        # Use the lower of the model's hard limit and the user's configured ceiling
        max_completion_tokens = min(model_max_completion_tokens, user_max)
    else:
        max_completion_tokens = user_max

    cache_key = _build_llm_client_cache_key(llm_config, max_completion_tokens)
    return _get_llm_client_cached(cache_key)

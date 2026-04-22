"""Get the LLM client."""

from enum import Enum
from typing import Optional

from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.ollama.adapter import (
    OllamaAPIAdapter,
)
from cognee.infrastructure.llm.exceptions import (
    LLMAPIKeyNotSetError,
    UnsupportedLLMProviderError,
)


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


# Config-keyed cache of instantiated LLM adapters. Callers hit
# ``get_llm_client`` on every ``acreate_structured_output`` invocation — and
# each fresh adapter re-builds per-instance instructor/pydantic state that
# pydantic's global SchemaValidator/Serializer dicts never release. Caching
# by config tuple means: same config → same adapter reused; config mutated
# via ``cognee.config.set(...)`` → cache key changes → new adapter. No
# explicit cache-clear needed.
_LLM_CLIENT_CACHE: dict = {}


def _secret_fingerprint(secret) -> Optional[str]:
    """Hash secrets so rotated keys partition the cache without ever storing
    the raw secret in the key tuple (which would surface in repr / crash
    dumps / logging of the global cache dict).
    """
    if secret is None or secret == "":
        return None
    import hashlib

    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()


def _llm_client_cache_key(llm_config, raise_api_key_error: bool) -> tuple:
    return (
        getattr(llm_config, "llm_provider", None),
        getattr(llm_config, "llm_model", None),
        getattr(llm_config, "llm_endpoint", None),
        getattr(llm_config, "llm_api_version", None),
        _secret_fingerprint(getattr(llm_config, "llm_api_key", None)),
        getattr(llm_config, "llm_max_completion_tokens", None),
        getattr(llm_config, "llm_instructor_mode", None),
        getattr(llm_config, "llm_streaming", None),
        getattr(llm_config, "fallback_model", None),
        _secret_fingerprint(getattr(llm_config, "fallback_api_key", None)),
        getattr(llm_config, "fallback_endpoint", None),
        bool(raise_api_key_error),
    )


def get_llm_client(raise_api_key_error: bool = True):
    """
    Get the LLM client based on the configuration using Enums.

    Clients are cached by a config tuple so repeated cognify calls don't
    re-mint Instructor+OpenAI clients. Each fresh adapter construction
    re-registers pydantic response-model validators that pydantic's global
    caches never release — observed as steady per-cycle FieldInfo /
    ModelMetaclass growth in long-running benchmark runs.
    """
    llm_config = get_llm_config()
    cache_key = _llm_client_cache_key(llm_config, raise_api_key_error)
    cached = _LLM_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    client = _build_llm_client(llm_config, raise_api_key_error)
    _LLM_CLIENT_CACHE[cache_key] = client
    return client


def _build_llm_client(llm_config, raise_api_key_error: bool):
    """Provider-dispatch body. See ``get_llm_client`` for the caching wrapper."""
    provider = LLMProvider(llm_config.llm_provider)

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

    llm_args = llm_config.llm_args

    if provider == LLMProvider.AZURE:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter import (
            AzureOpenAIAdapter,
        )

        return AzureOpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            streaming=llm_config.llm_streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
            llm_args=llm_args,
            use_managed_identity=llm_config.llm_azure_use_managed_identity,
        )

    elif provider == LLMProvider.OPENAI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.openai.adapter import (
            OpenAIAdapter,
        )

        return OpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            streaming=llm_config.llm_streaming,
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return OllamaAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Ollama",
            max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            llm_args=llm_args,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.anthropic.adapter import (
            AnthropicAdapter,
        )

        return AnthropicAdapter(
            llm_config.llm_api_key,
            llm_config.llm_model,
            max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            llm_args=llm_args,
        )

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return GenericAPIAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            name="Custom",
            endpoint=llm_config.llm_endpoint,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
            llm_args=llm_args,
        )

    elif provider == LLMProvider.GEMINI:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.gemini.adapter import (
            GeminiAdapter,
        )

        return GeminiAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            llm_args=llm_args,
        )

    elif provider == LLMProvider.MISTRAL:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.mistral.adapter import (
            MistralAdapter,
        )

        return MistralAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            endpoint=llm_config.llm_endpoint,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            llm_args=llm_args,
        )

    elif provider == LLMProvider.BEDROCK:
        # if llm_config.llm_api_key is None and raise_api_key_error:
        #     raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.bedrock.adapter import (
            BedrockAdapter,
        )

        return BedrockAdapter(
            model=llm_config.llm_model,
            api_key=llm_config.llm_api_key,
            max_completion_tokens=max_completion_tokens,
            streaming=llm_config.llm_streaming,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            llm_args=llm_args,
        )

    elif provider == LLMProvider.LLAMA_CPP:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter import (
            LlamaCppAPIAdapter,
        )

        # Get optional local mode parameters (will be None if not set)
        model_path = llm_config.llama_cpp_model_path
        n_ctx = llm_config.llama_cpp_n_ctx
        n_gpu_layers = llm_config.llama_cpp_n_gpu_layers
        chat_format = llm_config.llama_cpp_chat_format

        return LlamaCppAPIAdapter(
            model=llm_config.llm_model,
            max_completion_tokens=max_completion_tokens,
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            endpoint=llm_config.llm_endpoint,
            api_key=llm_config.llm_api_key,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            llm_args=llm_args,
        )
    else:
        raise UnsupportedLLMProviderError(provider)

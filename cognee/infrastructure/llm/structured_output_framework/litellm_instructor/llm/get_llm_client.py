"""Get the LLM client."""

from enum import Enum

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
    BEDROCK = "bedrock"
    LLAMA_CPP = "llama_cpp"


def get_llm_client(raise_api_key_error: bool = True):
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

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_completion_tokens,
    )  # imported here to avoid circular imports

    model_max_completion_tokens = get_model_max_completion_tokens(llm_config.llm_model)
    max_completion_tokens = (
        model_max_completion_tokens
        if model_max_completion_tokens
        else llm_config.llm_max_completion_tokens
    )

    if provider == LLMProvider.OPENAI:
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
        )

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None and raise_api_key_error:
            raise LLMAPIKeyNotSetError()

        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
            GenericAPIAdapter,
        )

        return GenericAPIAdapter(
            llm_config.llm_api_key,
            llm_config.llm_model,
            max_completion_tokens,
            "Custom",
            instructor_mode=llm_config.llm_instructor_mode.lower(),
            fallback_api_key=llm_config.fallback_api_key,
            fallback_endpoint=llm_config.fallback_endpoint,
            fallback_model=llm_config.fallback_model,
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
        )

    elif provider == LLMProvider.LLAMA_CPP:
        from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llama_cpp.adapter import (
            LlamaCppAPIAdapter,
        )

        # Get optional local mode parameters (will be None if not set)
        # TODO: refactor llm_config to include these parameters, currently they cannot be defined and defaults are used
        model_path = getattr(llm_config, "llama_cpp_model_path", None)
        n_ctx = getattr(llm_config, "llama_cpp_n_ctx", 2048)
        n_gpu_layers = getattr(llm_config, "llama_cpp_n_gpu_layers", 0)
        chat_format = getattr(llm_config, "llama_cpp_chat_format", "chatml")

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
        )
    else:
        raise UnsupportedLLMProviderError(provider)

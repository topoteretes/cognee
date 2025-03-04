"""Get the LLM client."""

from enum import Enum

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.ollama.adapter import OllamaAPIAdapter


# Define an Enum for LLM Providers
class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    ANTHROPIC_VERTEX = "anthropic_vertex"
    ANTHROPIC_BEDROCK = "anthropic_bedrock"
    CUSTOM = "custom"
    GEMINI = "gemini"


def get_llm_client():
    """Get the LLM client based on the configuration using Enums."""
    llm_config = get_llm_config()

    provider = LLMProvider(llm_config.llm_provider)

    # Check if max_token value is defined in liteLLM for given model
    # if not use value from cognee configuration
    from cognee.infrastructure.llm.utils import (
        get_model_max_tokens,
    )  # imported here to avoid circular imports

    model_max_tokens = get_model_max_tokens(llm_config.llm_model)
    max_tokens = model_max_tokens if model_max_tokens else llm_config.llm_max_tokens

    if provider == LLMProvider.OPENAI:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .openai.adapter import OpenAIAdapter

        return OpenAIAdapter(
            api_key=llm_config.llm_api_key,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            model=llm_config.llm_model,
            transcription_model=llm_config.transcription_model,
            max_tokens=max_tokens,
            streaming=llm_config.llm_streaming,
        )

    elif provider == LLMProvider.OLLAMA:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return OllamaAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Ollama",
            max_tokens=max_tokens,
        )

    elif provider == LLMProvider.ANTHROPIC:
        from .anthropic.adapter import AnthropicAdapter

        return AnthropicAdapter(max_tokens=max_tokens, model=llm_config.llm_model)
        
    elif provider == LLMProvider.ANTHROPIC_VERTEX:
        from .anthropic.vertex_adapter import AnthropicVertexAdapter

        # Check for Google Cloud specific configuration
        project_id = getattr(llm_config, "gcp_project_id", None)
        location = getattr(llm_config, "gcp_location", None)
        
        return AnthropicVertexAdapter(
            max_tokens=max_tokens, 
            model=llm_config.llm_model,
            project_id=project_id,
            location=location
        )
        
    elif provider == LLMProvider.ANTHROPIC_BEDROCK:
        from .anthropic.bedrock_adapter import AnthropicBedrockAdapter

        # Check for AWS specific configuration
        aws_profile = getattr(llm_config, "aws_profile", None)
        aws_region = getattr(llm_config, "aws_region", None)
        aws_access_key = getattr(llm_config, "aws_access_key", None)
        aws_secret_key = getattr(llm_config, "aws_secret_key", None)
        aws_session_token = getattr(llm_config, "aws_session_token", None)
        
        return AnthropicBedrockAdapter(
            max_tokens=max_tokens, 
            model=llm_config.llm_model,
            aws_profile=aws_profile,
            aws_region=aws_region,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token
        )

    elif provider == LLMProvider.CUSTOM:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .generic_llm_api.adapter import GenericAPIAdapter

        return GenericAPIAdapter(
            llm_config.llm_endpoint,
            llm_config.llm_api_key,
            llm_config.llm_model,
            "Custom",
            max_tokens=max_tokens,
        )

    elif provider == LLMProvider.GEMINI:
        if llm_config.llm_api_key is None:
            raise InvalidValueError(message="LLM API key is not set.")

        from .gemini.adapter import GeminiAdapter

        return GeminiAdapter(
            api_key=llm_config.llm_api_key,
            model=llm_config.llm_model,
            max_tokens=max_tokens,
            endpoint=llm_config.llm_endpoint,
            api_version=llm_config.llm_api_version,
            streaming=llm_config.llm_streaming,
        )

    else:
        raise InvalidValueError(message=f"Unsupported LLM provider: {provider}")

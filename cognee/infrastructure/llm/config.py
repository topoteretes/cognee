import os
from typing import Optional, ClassVar, Any
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator

try:
    from baml_py import ClientRegistry
except ImportError:
    ClientRegistry = None


class LLMConfig(BaseSettings):
    """
    Configuration settings for the LLM (Large Language Model) provider and related options.

    Public instance variables include:
    - llm_provider
    - llm_model
    - llm_endpoint
    - llm_api_key
    - llm_api_version
    - llm_temperature
    - llm_streaming
    - llm_max_completion_tokens
    - transcription_model
    - graph_prompt_path
    - llm_rate_limit_enabled
    - llm_rate_limit_requests
    - llm_rate_limit_interval
    - embedding_rate_limit_enabled
    - embedding_rate_limit_requests
    - embedding_rate_limit_interval

    Public methods include:
    - ensure_env_vars_for_ollama
    - to_dict
    """

    structured_output_framework: str = "instructor"
    llm_provider: str = "openai"
    llm_model: str = "openai/gpt-5-mini"
    llm_endpoint: str = ""
    llm_api_key: Optional[str] = None
    llm_api_version: Optional[str] = None
    llm_temperature: float = 0.0
    llm_streaming: bool = False
    llm_max_completion_tokens: int = 16384

    baml_llm_provider: str = "openai"
    baml_llm_model: str = "gpt-5-mini"
    baml_llm_endpoint: str = ""
    baml_llm_api_key: Optional[str] = None
    baml_llm_temperature: float = 0.0
    baml_llm_api_version: str = ""

    transcription_model: str = "whisper-1"
    graph_prompt_path: str = "generate_graph_prompt.txt"
    temporal_graph_prompt_path: str = "generate_event_graph_prompt.txt"
    event_entity_prompt_path: str = "generate_event_entity_prompt.txt"
    llm_rate_limit_enabled: bool = False
    llm_rate_limit_requests: int = 60
    llm_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)
    embedding_rate_limit_enabled: bool = False
    embedding_rate_limit_requests: int = 60
    embedding_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)

    fallback_api_key: str = ""
    fallback_endpoint: str = ""
    fallback_model: str = ""

    baml_registry: Optional[Any] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def model_post_init(self, __context) -> None:
        """Initialize the BAML registry after the model is created."""
        # Check if BAML is selected as structured output framework but not available
        if self.structured_output_framework.lower() == "baml" and ClientRegistry is None:
            raise ImportError(
                "BAML is selected as structured output framework but not available. "
                "Please install with 'pip install cognee\"[baml]\"' to use BAML extraction features."
            )
        elif self.structured_output_framework.lower() == "baml" and ClientRegistry is not None:
            self.baml_registry = ClientRegistry()

            raw_options = {
                "model": self.baml_llm_model,
                "temperature": self.baml_llm_temperature,
                "api_key": self.baml_llm_api_key,
                "base_url": self.baml_llm_endpoint,
                "api_version": self.baml_llm_api_version,
            }

            # Note: keep the item only when the value is not None or an empty string (they would override baml default values)
            options = {k: v for k, v in raw_options.items() if v not in (None, "")}
            self.baml_registry.add_llm_client(
                name=self.baml_llm_provider, provider=self.baml_llm_provider, options=options
            )
            # Sets the primary client
            self.baml_registry.set_primary(self.baml_llm_provider)

    @model_validator(mode="after")
    def ensure_env_vars_for_ollama(self) -> "LLMConfig":
        """
        Validate required environment variables for the 'ollama' LLM provider.

        Raises ValueError if some required environment variables are set without the others.
        Only checks are performed when 'llm_provider' is set to 'ollama'.

        Returns:
        --------

            - 'LLMConfig': The instance of LLMConfig after validation.
        """

        if self.llm_provider != "ollama":
            # Skip checks unless provider is "ollama"
            return self

        def is_env_set(var_name: str) -> bool:
            """
            Check if a given environment variable is set and non-empty.

            Parameters:
            -----------

                - var_name (str): The name of the environment variable to check.

            Returns:
            --------

                - bool: True if the environment variable exists and is not empty, otherwise False.
            """
            val = os.environ.get(var_name)
            return val is not None and val.strip() != ""

        #
        # 1. Check LLM environment variables
        #
        llm_env_vars = {
            "LLM_MODEL": is_env_set("LLM_MODEL"),
            "LLM_ENDPOINT": is_env_set("LLM_ENDPOINT"),
            "LLM_API_KEY": is_env_set("LLM_API_KEY"),
        }
        if any(llm_env_vars.values()) and not all(llm_env_vars.values()):
            missing_llm = [key for key, is_set in llm_env_vars.items() if not is_set]
            raise ValueError(
                "You have set some but not all of the required environment variables "
                f"for LLM usage (LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY). Missing: {missing_llm}"
            )

        #
        # 2. Check embedding environment variables
        #
        embedding_env_vars = {
            "EMBEDDING_PROVIDER": is_env_set("EMBEDDING_PROVIDER"),
            "EMBEDDING_MODEL": is_env_set("EMBEDDING_MODEL"),
            "EMBEDDING_DIMENSIONS": is_env_set("EMBEDDING_DIMENSIONS"),
            "HUGGINGFACE_TOKENIZER": is_env_set("HUGGINGFACE_TOKENIZER"),
        }
        if any(embedding_env_vars.values()) and not all(embedding_env_vars.values()):
            missing_embed = [key for key, is_set in embedding_env_vars.items() if not is_set]
            raise ValueError(
                "You have set some but not all of the required environment variables "
                "for embeddings (EMBEDDING_PROVIDER, EMBEDDING_MODEL, "
                "EMBEDDING_DIMENSIONS, HUGGINGFACE_TOKENIZER). Missing: "
                f"{missing_embed}"
            )

        return self

    def to_dict(self) -> dict:
        """
        Convert the LLMConfig instance into a dictionary representation.

        Returns:
        --------

            - dict: A dictionary containing the configuration settings of the LLMConfig
              instance.
        """
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "endpoint": self.llm_endpoint,
            "api_key": self.llm_api_key,
            "api_version": self.llm_api_version,
            "temperature": self.llm_temperature,
            "streaming": self.llm_streaming,
            "max_completion_tokens": self.llm_max_completion_tokens,
            "transcription_model": self.transcription_model,
            "graph_prompt_path": self.graph_prompt_path,
            "rate_limit_enabled": self.llm_rate_limit_enabled,
            "rate_limit_requests": self.llm_rate_limit_requests,
            "rate_limit_interval": self.llm_rate_limit_interval,
            "embedding_rate_limit_enabled": self.embedding_rate_limit_enabled,
            "embedding_rate_limit_requests": self.embedding_rate_limit_requests,
            "embedding_rate_limit_interval": self.embedding_rate_limit_interval,
            "fallback_api_key": self.fallback_api_key,
            "fallback_endpoint": self.fallback_endpoint,
            "fallback_model": self.fallback_model,
        }


@lru_cache
def get_llm_config():
    """
    Retrieve and cache the LLM configuration.

    This function returns an instance of the LLMConfig class. It leverages
    caching to ensure that repeated calls do not create new instances,
    but instead return the already created configuration object.

    Returns:
    --------

        - LLMConfig: An instance of the LLMConfig class containing the configuration for the
          LLM.
    """
    return LLMConfig()

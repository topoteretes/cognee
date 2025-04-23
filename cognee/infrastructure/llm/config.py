import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class LLMConfig(BaseSettings):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_endpoint: str = ""
    llm_api_key: Optional[str] = None
    llm_api_version: Optional[str] = None
    llm_temperature: float = 0.0
    llm_streaming: bool = False
    llm_max_tokens: int = 16384
    transcription_model: str = "whisper-1"
    graph_prompt_path: str = "generate_graph_prompt.txt"
    llm_rate_limit_enabled: bool = False
    llm_rate_limit_requests: int = 60
    llm_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)
    embedding_rate_limit_enabled: bool = False
    embedding_rate_limit_requests: int = 60
    embedding_rate_limit_interval: int = 60  # in seconds (default is 60 requests per minute)

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @model_validator(mode="after")
    def ensure_env_vars_for_ollama(self) -> "LLMConfig":
        """
        Only if llm_provider is 'ollama':
          - If any of (LLM_MODEL, LLM_ENDPOINT, LLM_API_KEY) is set, all must be set.
          - If any of (EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
            HUGGINGFACE_TOKENIZER) is set, all must be set.
        Otherwise, skip these checks.
        """

        if self.llm_provider != "ollama":
            # Skip checks unless provider is "ollama"
            return self

        def is_env_set(var_name: str) -> bool:
            """Return True if environment variable is present and non-empty."""
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
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "endpoint": self.llm_endpoint,
            "api_key": self.llm_api_key,
            "api_version": self.llm_api_version,
            "temperature": self.llm_temperature,
            "streaming": self.llm_streaming,
            "max_tokens": self.llm_max_tokens,
            "transcription_model": self.transcription_model,
            "graph_prompt_path": self.graph_prompt_path,
            "rate_limit_enabled": self.llm_rate_limit_enabled,
            "rate_limit_requests": self.llm_rate_limit_requests,
            "rate_limit_interval": self.llm_rate_limit_interval,
            "embedding_rate_limit_enabled": self.embedding_rate_limit_enabled,
            "embedding_rate_limit_requests": self.embedding_rate_limit_requests,
            "embedding_rate_limit_interval": self.embedding_rate_limit_interval,
        }


@lru_cache
def get_llm_config():
    return LLMConfig()

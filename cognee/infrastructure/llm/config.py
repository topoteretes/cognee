from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator, Field
import os

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

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @model_validator(mode="after")
    def ensure_all_or_none_from_env(self) -> "LLMConfig":
        """
        If any of (llm_model, llm_endpoint, llm_api_key) is explicitly set in the environment
        (including the .env file), make sure all are set.
        """

        def is_env_set(var_name: str) -> bool:
            """Return True if environment variable is present and non-empty."""
            val = os.environ.get(var_name)
            return val is not None and val.strip() != ""

        required_env_vars = {
            "LLM_MODEL": is_env_set("LLM_MODEL"),
            "LLM_ENDPOINT": is_env_set("LLM_ENDPOINT"),
            "LLM_API_KEY": is_env_set("LLM_API_KEY"),
        }

        # If at least one is set, ensure all are set
        if any(required_env_vars.values()) and not all(required_env_vars.values()):
            missing = [k for k, is_set in required_env_vars.items() if not is_set]
            raise ValueError(
                "You have set some but not all of the required environment variables "
                f"(`LLM_MODEL`, `LLM_ENDPOINT`, `LLM_API_KEY`). Missing: {missing}"
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
        }


@lru_cache
def get_llm_config():
    return LLMConfig()

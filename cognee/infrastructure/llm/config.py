from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


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

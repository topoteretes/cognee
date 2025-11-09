from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MinerUSettings(BaseSettings):
    """
    Settings for connecting to a remote MinerU HTTP service.

    Environment variables are namespaced with the ``MINERU_`` prefix.
    Example configuration:

        MINERU_ENABLED=true
        MINERU_SERVER_URL=http://mineru.example.com/v1
        MINERU_API_KEY=secret
        MINERU_API_KEY_HEADER=Authorization
    """

    enabled: bool = False
    server_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: str = "Authorization"
    max_completion_tokens: int = 2048
    timeout_seconds: int = 600
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    system_prompt: str = "You are an OCR assistant that produces faithful readings of document images."
    user_prompt: str = "Read the document image and return a clean transcription. Preserve tables using HTML when possible."
    detail: str = "high"

    model_config = SettingsConfigDict(env_file=".env", extra="allow", env_prefix="MINERU_")

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {self.api_key_header: self.api_key}

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.server_url)


@lru_cache
def get_mineru_settings() -> MinerUSettings:
    """
    Return cached MinerU settings instance.
    """

    return MinerUSettings()


from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseSettings):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_endpoint: str = ""
    llm_api_key: str = ""
    llm_streaming:bool = False

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "endpoint": self.llm_endpoint,
            "apiKey": self.llm_api_key,
        }

@lru_cache
def get_llm_config():
    return LLMConfig()

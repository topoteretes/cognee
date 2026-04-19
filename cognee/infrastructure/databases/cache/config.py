from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional, Literal


class CacheConfig(BaseSettings):
    """
    Configuration for distributed cache systems (e.g., Redis), used for locking or coordination.

    Attributes:
    - shared_kuzu_lock: Shared kuzu lock logic on/off.
    - cache_host: Hostname of the cache service.
    - cache_port: Port number for the cache service.
    - agentic_lock_expire: Automatic lock expiration time (in seconds).
    - agentic_lock_timeout: Maximum time (in seconds) to wait for the lock release.
    - session_ttl_seconds: Time-to-live for Redis session keys in seconds (default: 7 days).
      Positive values enable expiry; 0/None disables expiry.
    - usage_logging: Enable/disable usage logging for API endpoints and MCP tools.
    - usage_logging_ttl: Time-to-live for usage logs in seconds (default: 7 days).
    - auto_feedback: When caching is True, run automatic feedback detection on each query (default False).
    """

    cache_backend: Literal["redis", "fs", "tapes"] = "fs"
    caching: bool = True
    auto_feedback: bool = False
    shared_kuzu_lock: bool = False
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_username: Optional[str] = None
    cache_password: Optional[str] = None
    agentic_lock_expire: int = 240
    agentic_lock_timeout: int = 300
    session_ttl_seconds: Optional[int] = 604800
    max_session_context_chars: Optional[int] = None
    usage_logging: bool = False
    usage_logging_ttl: int = 604800
    tapes_ingest_url: str = "http://localhost:8082"
    tapes_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    tapes_agent_name: str = "cognee"
    tapes_model: str = "cognee-session"
    tapes_request_timeout: float = 5.0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "cache_backend": self.cache_backend,
            "caching": self.caching,
            "auto_feedback": self.auto_feedback,
            "shared_kuzu_lock": self.shared_kuzu_lock,
            "cache_host": self.cache_host,
            "cache_port": self.cache_port,
            "cache_username": self.cache_username,
            "cache_password": self.cache_password,
            "agentic_lock_expire": self.agentic_lock_expire,
            "agentic_lock_timeout": self.agentic_lock_timeout,
            "session_ttl_seconds": self.session_ttl_seconds,
            "max_session_context_chars": self.max_session_context_chars,
            "usage_logging": self.usage_logging,
            "usage_logging_ttl": self.usage_logging_ttl,
            "tapes_ingest_url": self.tapes_ingest_url,
            "tapes_provider": self.tapes_provider,
            "tapes_agent_name": self.tapes_agent_name,
            "tapes_model": self.tapes_model,
            "tapes_request_timeout": self.tapes_request_timeout,
        }


@lru_cache
def get_cache_config():
    return CacheConfig()

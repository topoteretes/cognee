from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class CacheConfig(BaseSettings):
    """
    Configuration for distributed cache systems (e.g., Redis), used for locking or coordination.

    Attributes:
    - shared_kuzu_lock: Shared kuzu lock logic on/off.
    - cache_host: Hostname of the cache service.
    - cache_port: Port number for the cache service.
    - agentic_lock_expire: Automatic lock expiration time (in seconds).
    - agentic_lock_timeout: Maximum time (in seconds) to wait for the lock release.
    """

    caching: bool = False
    shared_kuzu_lock: bool = False
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_username: Optional[str] = None
    cache_password: Optional[str] = None
    agentic_lock_expire: int = 240
    agentic_lock_timeout: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "caching": self.caching,
            "shared_kuzu_lock": self.shared_kuzu_lock,
            "cache_host": self.cache_host,
            "cache_port": self.cache_port,
            "cache_username": self.cache_username,
            "cache_password": self.cache_password,
            "agentic_lock_expire": self.agentic_lock_expire,
            "agentic_lock_timeout": self.agentic_lock_timeout,
        }


@lru_cache
def get_cache_config():
    return CacheConfig()

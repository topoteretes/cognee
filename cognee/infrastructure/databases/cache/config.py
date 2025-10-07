from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class CacheConfig(BaseSettings):
    """
    Configuration for distributed cache systems (e.g., Redis), used for locking or coordination.

    Attributes:
    - caching: Caching logic on/off.
    - cache_host: Hostname of the cache service.
    - cache_port: Port number for the cache service.
    - agentic_lock_expire: Automatic lock expiration time (in seconds).
    - agentic_lock_timeout: Maximum time (in seconds) to wait for the lock release.
    """

    caching: bool = False
    cache_host: str = "localhost"
    cache_port: int = 6379
    agentic_lock_expire: int = 240
    agentic_lock_timeout: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "caching": self.caching,
            "cache_host": self.cache_host,
            "cache_port": self.cache_port,
            "agentic_lock_expire": self.agentic_lock_expire,
            "agentic_lock_timeout": self.agentic_lock_timeout,
        }


@lru_cache
def get_cache_config():
    return CacheConfig()

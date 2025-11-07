from functools import lru_cache
from typing import Optional

from cognee.config.mineru import get_mineru_settings, MinerUSettings

from .http_client import MineruHTTPClient


@lru_cache
def get_mineru_http_client() -> Optional[MineruHTTPClient]:
    """
    Lazily create and cache a MinerU HTTP client if configuration is present.
    """

    settings = get_mineru_settings()
    if not settings.is_configured:
        return None

    return MineruHTTPClient(settings)


__all__ = ["get_mineru_http_client", "MineruHTTPClient", "get_mineru_settings", "MinerUSettings"]


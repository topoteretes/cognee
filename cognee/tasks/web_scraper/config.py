from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, Optional, Literal
import os


class WebScraperConfig(BaseSettings):
    tavily_api_key: Optional[str] = None
    web_scraper_timeout: float = 15.0
    web_scraper_max_delay: Optional[float] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "tavily_api_key": self.tavily_api_key,
            "web_scraper_timeout": self.web_scraper_timeout,
            "web_scraper_max_delay": self.web_scraper_max_delay,
        }


@lru_cache
def get_web_scraper_config():
    return WebScraperConfig()


class TavilyConfig(BaseModel):
    api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    extract_depth: Literal["basic", "advanced"] = "basic"
    proxies: Optional[Dict[str, str]] = None
    timeout: Optional[int] = Field(default=10, ge=1, le=60)


class DefaultCrawlerConfig(BaseModel):
    concurrency: int = 5
    crawl_delay: float = 0.5
    max_crawl_delay: Optional[float] = (
        10.0  # Maximum crawl delay to respect from robots.txt (None = no limit)
    )
    timeout: float = float(os.getenv("WEB_SCRAPER_TIMEOUT", 15.0))
    max_retries: int = 2
    retry_delay_factor: float = 0.5
    headers: Optional[Dict[str, str]] = None
    use_playwright: bool = False
    playwright_js_wait: float = 0.8
    robots_cache_ttl: float = 3600.0
    join_all_matches: bool = False

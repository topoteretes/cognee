from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Literal
import os


class TavilyConfig(BaseModel):
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    extract_depth: Literal["basic", "advanced"] = "basic"
    proxies: Optional[Dict[str, str]] = None
    timeout: Optional[int] = Field(
        default_factory=lambda: int(os.getenv("TAVILY_TIMEOUT", "10")), ge=1, le=60
    )


class DefaultCrawlerConfig(BaseModel):
    concurrency: int = Field(default_factory=lambda: int(os.getenv("WEB_SCRAPER_CONCURRENCY", "5")))
    crawl_delay: float = Field(
        default_factory=lambda: float(os.getenv("WEB_SCRAPER_CRAWL_DELAY", "0.5"))
    )
    max_crawl_delay: Optional[float] = Field(
        default_factory=lambda: float(
            os.getenv("WEB_SCRAPER_MAX_DELAY", "10.0")
        )  # Maximum crawl delay to respect from robots.txt (None = no limit)
    )
    timeout: float = Field(default_factory=lambda: float(os.getenv("WEB_SCRAPER_TIMEOUT", "15.0")))
    max_retries: int = 2
    retry_delay_factor: float = 0.5
    headers: Optional[Dict[str, str]] = None
    use_playwright: bool = False
    playwright_js_wait: float = 0.8
    robots_cache_ttl: float = 3600.0
    join_all_matches: bool = False

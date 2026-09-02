from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal
import os


class TavilyConfig(BaseModel):
    api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    extract_depth: Literal["basic", "advanced"] = "basic"
    proxies: Optional[Dict[str, str]] = None
    timeout: Optional[int] = Field(default=10, ge=1, le=60)


class KeenableConfig(BaseModel):
    api_key: Optional[str] = os.getenv("KEENABLE_API_KEY")
    base_url: str = os.getenv("KEENABLE_BASE_URL", "https://api.keenable.ai")
    live: bool = os.getenv("KEENABLE_LIVE_FETCH", "false").lower() in ("true", "1", "yes")
    prompt: Optional[str] = Field(default=None, max_length=2000)
    concurrency: int = Field(default=5, ge=1)
    timeout: Optional[int] = Field(default=30, ge=1, le=120)


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

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
import os


class TavilyConfig(BaseModel):
    api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    extract_depth: Literal["basic", "advanced"] = "basic"
    proxies: Optional[Dict[str, str]] = None
    timeout: Optional[int] = Field(default=10, ge=1, le=60)


class ExaConfig(BaseModel):
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("EXA_API_KEY"))
    num_results: int = Field(default=10, ge=1, le=100)
    search_type: Literal["neural", "fast", "auto"] = "auto"
    use_highlights: bool = True
    highlights_max_characters: int = Field(default=500, ge=1)
    use_text: bool = True
    text_max_characters: int = Field(default=1000, ge=1)
    use_summary: bool = False
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    include_text: Optional[List[str]] = None
    exclude_text: Optional[List[str]] = None
    category: Optional[
        Literal["company", "research paper", "news", "personal site", "financial report", "people"]
    ] = None
    start_published_date: Optional[str] = None
    end_published_date: Optional[str] = None


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

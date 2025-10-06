from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Literal
import os


class TavilyConfig(BaseModel):
    api_key: str = os.getenv("TAVILY_API_KEY")
    extract_depth: str = "basic"
    timeout: int = Field(None, ge=1, le=60)


class SoupCrawlerConfig(BaseModel):
    concurrency: int = 5
    crawl_delay: float = 0.5
    timeout: float = 15.0
    max_retries: int = 2
    retry_delay_factor: float = 0.5
    headers: Optional[Dict[str, str]] = None
    extraction_rules: Dict[str, Any]
    use_playwright: bool = False
    playwright_js_wait: float = 0.8
    join_all_matches: bool = False

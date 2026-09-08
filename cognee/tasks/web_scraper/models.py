from datetime import datetime
from typing import Any, Dict, List, Optional

from cognee.infrastructure.engine import DataPoint


class WebPage(DataPoint):
    """Represents a scraped web page with metadata"""

    name: str | None
    content: str
    content_hash: str
    scraped_at: datetime
    last_modified: datetime | None
    status_code: int
    content_type: str
    page_size: int
    extraction_rules: dict[str, Any]  # CSS selectors, XPath rules used
    description: str
    metadata: dict = {"index_fields": ["name", "description", "content"]}


class WebSite(DataPoint):
    """Represents a website or domain being scraped"""

    name: str
    base_url: str
    robots_txt: str | None
    crawl_delay: float
    last_crawled: datetime
    page_count: int
    scraping_config: dict[str, Any]
    description: str
    metadata: dict = {"index_fields": ["name", "description"]}


class ScrapingJob(DataPoint):
    """Represents a scraping job configuration"""

    name: str
    urls: list[str]
    schedule: str | None  # Cron-like schedule for recurring scrapes
    status: str  # "active", "paused", "completed", "failed"
    last_run: datetime | None
    next_run: datetime | None
    description: str
    metadata: dict = {"index_fields": ["name", "description"]}

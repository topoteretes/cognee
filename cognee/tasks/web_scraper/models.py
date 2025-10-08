from cognee.infrastructure.engine import DataPoint
from typing import Optional, Dict, Any, List
from datetime import datetime


class WebPage(DataPoint):
    """Represents a scraped web page with metadata"""

    name: Optional[str]
    content: str
    content_hash: str
    scraped_at: datetime
    last_modified: Optional[datetime]
    status_code: int
    content_type: str
    page_size: int
    extraction_rules: Dict[str, Any]  # CSS selectors, XPath rules used
    description: str
    metadata: dict = {"index_fields": ["name", "description", "content"]}


class WebSite(DataPoint):
    """Represents a website or domain being scraped"""

    name: str
    base_url: str
    robots_txt: Optional[str]
    crawl_delay: float
    last_crawled: datetime
    page_count: int
    scraping_config: Dict[str, Any]
    description: str
    metadata: dict = {"index_fields": ["name", "description"]}


class ScrapingJob(DataPoint):
    """Represents a scraping job configuration"""

    name: str
    urls: List[str]
    schedule: Optional[str]  # Cron-like schedule for recurring scrapes
    status: str  # "active", "paused", "completed", "failed"
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    description: str
    metadata: dict = {"index_fields": ["name", "description"]}

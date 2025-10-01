from cognee.infrastructure.engine.models import DataPoint
from typing import Optional, Dict, Any, List
from datetime import datetime


class WebPage(DataPoint):
    """Represents a scraped web page with metadata"""

    url: str
    title: Optional[str]
    content: str
    content_hash: str
    scraped_at: datetime
    last_modified: Optional[datetime]
    status_code: int
    content_type: str
    page_size: int
    extraction_rules: Dict[str, Any]  # CSS selectors, XPath rules used
    metadata: dict = {"index_fields": ["url", "title", "scraped_at"]}


class WebSite(DataPoint):
    """Represents a website or domain being scraped"""

    domain: str
    base_url: str
    robots_txt: Optional[str]
    crawl_delay: float
    last_crawled: datetime
    page_count: int
    scraping_config: Dict[str, Any]
    metadata: dict = {"index_fields": ["domain", "base_url"]}


class ScrapingJob(DataPoint):
    """Represents a scraping job configuration"""

    job_name: str
    urls: List[str]
    schedule: Optional[str]  # Cron-like schedule for recurring scrapes
    status: str  # "active", "paused", "completed", "failed"
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    metadata: dict = {"index_fields": ["job_name", "status"]}

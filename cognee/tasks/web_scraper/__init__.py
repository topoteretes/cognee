"""Web scraping module for cognee.

This module provides tools for scraping web content, managing scraping jobs, and storing
data in a graph database. It includes classes and functions for crawling web pages using
BeautifulSoup or Tavily, defining data models, and handling scraping configurations.
"""

from .utils import fetch_page_content
from .default_url_crawler import DefaultUrlCrawler

# Lazy import for web_scraper_task to avoid requiring apscheduler
# Import these directly if needed: from cognee.tasks.web_scraper.web_scraper_task import ...


def __getattr__(name):
    """Lazy load web scraper task functions that require apscheduler."""
    if name == "cron_web_scraper_task":
        from .web_scraper_task import cron_web_scraper_task

        return cron_web_scraper_task
    elif name == "web_scraper_task":
        from .web_scraper_task import web_scraper_task

        return web_scraper_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BeautifulSoupCrawler",
    "fetch_page_content",
    "cron_web_scraper_task",
    "web_scraper_task",
    "DefaultUrlCrawler",
]

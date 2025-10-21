"""Web scraping module for cognee.

This module provides tools for scraping web content, managing scraping jobs, and storing
data in a graph database. It includes classes and functions for crawling web pages using
BeautifulSoup or Tavily, defining data models, and handling scraping configurations.
"""

from .bs4_crawler import BeautifulSoupCrawler
from .utils import fetch_page_content
from .web_scraper_task import cron_web_scraper_task, web_scraper_task
from .default_url_crawler import DefaultUrlCrawler


__all__ = [
    "BeautifulSoupCrawler",
    "fetch_page_content",
    "cron_web_scraper_task",
    "web_scraper_task",
    "DefaultUrlCrawler",
]

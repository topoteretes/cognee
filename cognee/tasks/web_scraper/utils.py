"""Utilities for fetching web content using BeautifulSoup or Tavily.

This module provides functions to fetch and extract content from web pages, supporting
both BeautifulSoup for custom extraction rules and Tavily for API-based scraping.
"""

from typing import Dict, List, Union, Optional, Literal
from cognee.shared.logging_utils import get_logger
from .bs4_crawler import BeautifulSoupCrawler
from .config import TavilyConfig, SoupCrawlerConfig

logger = get_logger(__name__)


async def fetch_page_content(
    urls: Union[str, List[str]],
    *,
    preferred_tool: Optional[Literal["tavily", "beautifulsoup"]] = "beautifulsoup",
    tavily_config: Optional[TavilyConfig] = None,
    soup_crawler_config: Optional[SoupCrawlerConfig] = None,
) -> Dict[str, str]:
    """Fetch content from one or more URLs using the specified tool.

    This function retrieves web page content using either BeautifulSoup (with custom
    extraction rules) or Tavily (API-based scraping). It handles single URLs or lists of
    URLs and returns a dictionary mapping URLs to their extracted content.

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to scrape.
        preferred_tool: The scraping tool to use ("tavily" or "beautifulsoup").
            Defaults to "beautifulsoup".
        tavily_config: Configuration for Tavily API, including API key.
            Required if preferred_tool is "tavily".
        soup_crawler_config: Configuration for BeautifulSoup crawler, including
            extraction rules. Required if preferred_tool is "beautifulsoup" and
            extraction_rules are needed.

    Returns:
        Dict[str, str]: A dictionary mapping each URL to its
            extracted content (as a string for BeautifulSoup or a dict for Tavily).

    Raises:
        ValueError: If Tavily API key is missing when using Tavily, or if
            extraction_rules are not provided when using BeautifulSoup.
        ImportError: If required dependencies (beautifulsoup4 or tavily-python) are not
            installed.
    """
    if preferred_tool == "tavily":
        if not tavily_config or tavily_config.api_key is None:
            raise ValueError("TAVILY_API_KEY must be set in TavilyConfig to use Tavily")
        return await fetch_with_tavily(urls, tavily_config)

    if preferred_tool == "beautifulsoup":
        try:
            from bs4 import BeautifulSoup as _  # noqa: F401
        except ImportError:
            logger.error(
                "Failed to import bs4, make sure to install using pip install beautifulsoup4>=4.13.1"
            )
            raise ImportError
        if not soup_crawler_config or soup_crawler_config.extraction_rules is None:
            raise ValueError("extraction_rules must be provided when not using Tavily")
        extraction_rules = soup_crawler_config.extraction_rules
        crawler = BeautifulSoupCrawler(
            concurrency=soup_crawler_config.concurrency,
            crawl_delay=soup_crawler_config.crawl_delay,
            timeout=soup_crawler_config.timeout,
            max_retries=soup_crawler_config.max_retries,
            retry_delay_factor=soup_crawler_config.retry_delay_factor,
            headers=soup_crawler_config.headers,
        )
        try:
            results = await crawler.fetch_with_bs4(
                urls,
                extraction_rules,
                use_playwright=soup_crawler_config.use_playwright,
                playwright_js_wait=soup_crawler_config.playwright_js_wait,
                join_all_matches=soup_crawler_config.join_all_matches,
            )
            return results
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")
            raise


async def fetch_with_tavily(
    urls: Union[str, List[str]], tavily_config: Optional[TavilyConfig] = None
) -> Dict[str, str]:
    """Fetch content from URLs using the Tavily API.

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to scrape.

    Returns:
        Dict[str, str]: A dictionary mapping each URL to its raw content as a string.

    Raises:
        ImportError: If tavily-python is not installed.
        Exception: If the Tavily API request fails.
    """
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        logger.error(
            "Failed to import tavily, make sure to install using pip install tavily-python>=0.7.0"
        )
        raise
    client = AsyncTavilyClient(api_key=tavily_config.api_key if tavily_config else None)
    results = await client.extract(urls, format="text")
    for failed_result in results.get("failed_results", []):
        logger.warning(f"Failed to fetch {failed_result}")
    return_results = {}
    for result in results.get("results", []):
        return_results[result["url"]] = result["raw_content"]
    return return_results

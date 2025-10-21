"""Utilities for fetching web content using BeautifulSoup or Tavily.

This module provides functions to fetch and extract content from web pages, supporting
both BeautifulSoup for custom extraction rules and Tavily for API-based scraping.
"""

from typing import Dict, List, Union, Optional, Literal
from cognee.context_global_variables import soup_crawler_config, tavily_config
from cognee.shared.logging_utils import get_logger
from .bs4_crawler import BeautifulSoupCrawler
from .config import TavilyConfig

logger = get_logger(__name__)


async def fetch_page_content(
    urls: Union[str, List[str]],
    preferred_tool: Optional[Literal["tavily", "beautifulsoup"]] = "beautifulsoup",
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
    url_list = [urls] if isinstance(urls, str) else urls
    logger.info(f"Starting to fetch content from {len(url_list)} URL(s) using {preferred_tool}")

    _tavily_config = tavily_config.get()
    _soup_crawler_config = soup_crawler_config.get()

    if preferred_tool == "tavily":
        if not tavily_config or tavily_config.api_key is None:
            raise ValueError("TAVILY_API_KEY must be set in TavilyConfig to use Tavily")
        logger.info("Using Tavily API for content extraction")
        return await fetch_with_tavily(urls, tavily_config)

    if preferred_tool == "beautifulsoup":
        try:
            from bs4 import BeautifulSoup as _  # noqa: F401
        except ImportError:
            logger.error(
                "Failed to import bs4, make sure to install using pip install beautifulsoup4>=4.13.1"
            )
            raise ImportError
        if soup_crawler_config is None or soup_crawler_config.extraction_rules is None:
            raise ValueError("soup_crawler_config must be provided when not using Tavily")

        logger.info("Using BeautifulSoup for content extraction")
        logger.info(
            f"Initializing BeautifulSoup crawler with concurrency={soup_crawler_config.concurrency}, timeout={soup_crawler_config.timeout}s, max_crawl_delay={soup_crawler_config.max_crawl_delay}s"
        )

        crawler = BeautifulSoupCrawler(
            concurrency=soup_crawler_config.concurrency,
            crawl_delay=soup_crawler_config.crawl_delay,
            max_crawl_delay=soup_crawler_config.max_crawl_delay,
            timeout=soup_crawler_config.timeout,
            max_retries=soup_crawler_config.max_retries,
            retry_delay_factor=soup_crawler_config.retry_delay_factor,
            headers=soup_crawler_config.headers,
            robots_cache_ttl=soup_crawler_config.robots_cache_ttl,
        )
        try:
            logger.info(
                f"Starting to crawl {len(url_list)} URL(s) with BeautifulSoup (use_playwright={soup_crawler_config.use_playwright})"
            )
            results = await crawler.fetch_urls(
                urls,
                use_playwright=soup_crawler_config.use_playwright,
                playwright_js_wait=soup_crawler_config.playwright_js_wait,
            )
            logger.info(f"Successfully fetched content from {len(results)} URL(s)")
            return results
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")
            raise
        finally:
            logger.info("Closing BeautifulSoup crawler")
            await crawler.close()


async def fetch_with_tavily(
    urls: Union[str, List[str]], tavily_config: TavilyConfig
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

    url_list = [urls] if isinstance(urls, str) else urls
    extract_depth = tavily_config.extract_depth if tavily_config else "basic"
    timeout = tavily_config.timeout if tavily_config else 10

    logger.info(
        f"Initializing Tavily client with extract_depth={extract_depth}, timeout={timeout}s"
    )
    client = AsyncTavilyClient(
        api_key=tavily_config.api_key,
        proxies=tavily_config.proxies,
    )

    logger.info(f"Sending extract request to Tavily API for {len(url_list)} URL(s)")
    results = await client.extract(
        urls,
        format="text",
        extract_depth=extract_depth,
        timeout=timeout,
    )

    failed_count = len(results.get("failed_results", []))
    if failed_count > 0:
        logger.warning(f"Tavily API failed to fetch {failed_count} URL(s)")
        for failed_result in results.get("failed_results", []):
            logger.warning(f"Failed to fetch {failed_result}")

    return_results = {}
    for result in results.get("results", []):
        return_results[result["url"]] = result["raw_content"]

    logger.info(f"Successfully fetched content from {len(return_results)} URL(s) via Tavily")
    return return_results

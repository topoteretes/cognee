"""Utilities for fetching web content using BeautifulSoup, Tavily, or Keenable.

This module provides functions to fetch and extract content from web pages, supporting
BeautifulSoup for custom extraction rules and Tavily or Keenable for API-based scraping.
"""

import asyncio
import os
from typing import List, Optional, Union

import httpx
from cognee.shared.logging_utils import get_logger
from cognee.tasks.web_scraper.types import UrlsToHtmls
from .default_url_crawler import DefaultUrlCrawler
from .config import DefaultCrawlerConfig, KeenableConfig, TavilyConfig

logger = get_logger(__name__)


async def fetch_page_content(
    urls: Union[str, List[str]],
    preferred_tool: Optional[str] = None,
    tavily_config: Optional[TavilyConfig] = None,
    keenable_config: Optional[KeenableConfig] = None,
    soup_crawler_config: Optional[DefaultCrawlerConfig] = None,
) -> UrlsToHtmls:
    """Fetch content from one or more URLs using the specified tool.

    This function retrieves web page content using BeautifulSoup (with custom
    extraction rules), Tavily, or Keenable (API-based scraping). It handles single URLs
    or lists of URLs and returns a dictionary mapping URLs to their extracted content.

    When preferred_tool is not given, the backend is auto-selected from the
    environment: Tavily if TAVILY_API_KEY is set, otherwise Keenable if
    KEENABLE_API_KEY is set, otherwise the default crawler.

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to scrape.
        preferred_tool: The scraping tool to use ("tavily", "keenable", or
            "beautifulsoup"). Defaults to environment-based auto-selection.
        tavily_config: Configuration for Tavily API, including API key.
        keenable_config: Configuration for the Keenable API.
        soup_crawler_config: Configuration for the default (BeautifulSoup) crawler.

    Returns:
        Dict[str, str]: A dictionary mapping each URL to its extracted content.

    Raises:
        ValueError: If an unknown preferred_tool is given.
        ImportError: If required dependencies (e.g., tavily-python) are not installed.
    """
    url_list = [urls] if isinstance(urls, str) else urls

    if preferred_tool is None:
        if os.getenv("TAVILY_API_KEY"):
            preferred_tool = "tavily"
        elif os.getenv("KEENABLE_API_KEY"):
            preferred_tool = "keenable"
        else:
            preferred_tool = "beautifulsoup"

    if preferred_tool == "tavily":
        logger.info("Using Tavily API for url fetching")
        return await fetch_with_tavily(urls, tavily_config=tavily_config)
    elif preferred_tool == "keenable":
        logger.info("Using Keenable API for url fetching")
        return await fetch_with_keenable(urls, keenable_config=keenable_config)
    elif preferred_tool == "beautifulsoup":
        logger.info("Using default crawler for content extraction")

        default_crawler_config = soup_crawler_config or (
            DefaultCrawlerConfig()
        )  # We've decided to use defaults, and configure through env vars as needed

        logger.info(
            f"Initializing BeautifulSoup crawler with concurrency={default_crawler_config.concurrency}, timeout={default_crawler_config.timeout}s, max_crawl_delay={default_crawler_config.max_crawl_delay}s"
        )

        crawler = DefaultUrlCrawler(
            concurrency=default_crawler_config.concurrency,
            crawl_delay=default_crawler_config.crawl_delay,
            max_crawl_delay=default_crawler_config.max_crawl_delay,
            timeout=default_crawler_config.timeout,
            max_retries=default_crawler_config.max_retries,
            retry_delay_factor=default_crawler_config.retry_delay_factor,
            headers=default_crawler_config.headers,
            robots_cache_ttl=default_crawler_config.robots_cache_ttl,
        )
        try:
            logger.info(
                f"Starting to crawl {len(url_list)} URL(s) with BeautifulSoup (use_playwright={default_crawler_config.use_playwright})"
            )
            results = await crawler.fetch_urls(
                urls,
                use_playwright=default_crawler_config.use_playwright,
                playwright_js_wait=default_crawler_config.playwright_js_wait,
            )
            logger.info(f"Successfully fetched content from {len(results)} URL(s)")
            return results
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")
            raise
        finally:
            logger.info("Closing BeautifulSoup crawler")
            await crawler.close()
    else:
        raise ValueError(
            f"Unknown preferred_tool '{preferred_tool}'. "
            "Expected 'tavily', 'keenable', or 'beautifulsoup'."
        )


async def fetch_with_tavily(
    urls: Union[str, List[str]], tavily_config: Optional[TavilyConfig] = None
) -> UrlsToHtmls:
    """Fetch content from URLs using the Tavily API.

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to scrape.
        tavily_config: Configuration for the Tavily API. Defaults to environment-based
            configuration (TAVILY_API_KEY).

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

    tavily_config = tavily_config or TavilyConfig()
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


async def fetch_with_keenable(
    urls: Union[str, List[str]], keenable_config: Optional[KeenableConfig] = None
) -> UrlsToHtmls:
    """Fetch content from URLs using the Keenable API.

    Uses Keenable's /v1/fetch endpoint (https://docs.keenable.ai) to extract clean
    markdown content from each URL. Requires an API key (KEENABLE_API_KEY or
    keenable_config.api_key).

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to scrape.
        keenable_config: Configuration for the Keenable API. Defaults to
            environment-based configuration (KEENABLE_API_KEY, KEENABLE_BASE_URL).

    Returns:
        Dict[str, str]: A dictionary mapping each requested URL to its extracted
            content. URLs that fail to fetch are omitted with a warning.

    Raises:
        Exception: If every requested URL fails to fetch (e.g., invalid API key).
    """
    keenable_config = keenable_config or KeenableConfig()
    url_list = [urls] if isinstance(urls, str) else urls

    headers = {}
    if keenable_config.api_key:
        headers["X-API-Key"] = keenable_config.api_key

    logger.info(
        f"Sending fetch requests to Keenable API for {len(url_list)} URL(s) "
        f"(live={keenable_config.live}, timeout={keenable_config.timeout}s)"
    )

    semaphore = asyncio.Semaphore(keenable_config.concurrency)

    async with httpx.AsyncClient(
        base_url=keenable_config.base_url,
        headers=headers,
        timeout=keenable_config.timeout,
    ) as client:

        async def fetch_one(url: str):
            params = {"url": url}
            if keenable_config.live:
                params["live"] = "true"
            if keenable_config.prompt:
                params["prompt"] = keenable_config.prompt
            async with semaphore:
                response = await client.get("/v1/fetch", params=params)
            response.raise_for_status()
            return response.json().get("content")

        responses = await asyncio.gather(
            *(fetch_one(url) for url in url_list), return_exceptions=True
        )

    return_results = {}
    errors = []
    # URLs and error details can embed credentials — log only positions and
    # exception class names.
    for position, (url, result) in enumerate(zip(url_list, responses), start=1):
        if isinstance(result, BaseException):
            logger.warning(
                "Keenable API failed to fetch URL %d of %d (%s)",
                position,
                len(url_list),
                type(result).__name__,
            )
            errors.append(result)
        elif result is None:
            logger.warning(
                "Keenable API returned no content for URL %d of %d", position, len(url_list)
            )
        else:
            return_results[url] = result

    if url_list and not return_results and errors:
        raise errors[0]

    logger.info(f"Successfully fetched content from {len(return_results)} URL(s) via Keenable")
    return return_results

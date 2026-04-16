"""Utilities for fetching web content using BeautifulSoup, Tavily, or Exa.

This module provides functions to fetch and extract content from web pages, supporting
BeautifulSoup for custom extraction rules, Tavily for API-based scraping, and Exa for
AI-powered web search and content retrieval.
"""

import asyncio
import os
from typing import Dict, List, Optional, Union
from cognee.shared.logging_utils import get_logger
from cognee.tasks.web_scraper.types import UrlsToHtmls
from .default_url_crawler import DefaultUrlCrawler
from .config import DefaultCrawlerConfig, ExaConfig, TavilyConfig

logger = get_logger(__name__)


async def fetch_page_content(urls: Union[str, List[str]]) -> UrlsToHtmls:
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
        default_crawler_config: Configuration for BeautifulSoup crawler, including
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

    if os.getenv("TAVILY_API_KEY"):
        logger.info("Using Tavily API for url fetching")
        return await fetch_with_tavily(urls)
    else:
        logger.info("Using default crawler for content extraction")

        default_crawler_config = (
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


async def fetch_with_exa(urls: Union[str, List[str]], exa_config: Optional[ExaConfig] = None) -> UrlsToHtmls:
    """Fetch content from URLs using the Exa API's get_contents endpoint.

    Args:
        urls: A single URL (str) or a list of URLs (List[str]) to fetch content from.
        exa_config: Configuration for Exa API. If None, defaults are used.

    Returns:
        Dict[str, str]: A dictionary mapping each URL to its extracted content.

    Raises:
        ImportError: If exa-py is not installed.
        Exception: If the Exa API request fails.
    """
    try:
        from exa_py import Exa
    except ImportError:
        logger.error(
            "Failed to import exa_py, make sure to install using pip install exa-py>=2.0.0"
        )
        raise

    config = exa_config or ExaConfig()
    url_list = [urls] if isinstance(urls, str) else urls

    logger.info(f"Initializing Exa client for content fetching of {len(url_list)} URL(s)")
    client = Exa(api_key=config.api_key)
    client.headers["x-exa-integration"] = "cognee"

    contents_kwargs: Dict = {}
    if config.use_text:
        contents_kwargs["text"] = {"max_characters": config.text_max_characters}
    if config.use_highlights:
        contents_kwargs["highlights"] = {"max_characters": config.highlights_max_characters}
    if config.use_summary:
        contents_kwargs["summary"] = True

    logger.info(f"Sending get_contents request to Exa API for {len(url_list)} URL(s)")
    response = await asyncio.to_thread(client.get_contents, url_list, **contents_kwargs)

    return_results: Dict[str, str] = {}
    for result in response.results:
        url = getattr(result, "url", "") or ""
        content = _extract_exa_content(result)
        if url:
            return_results[url] = content

    logger.info(f"Successfully fetched content from {len(return_results)} URL(s) via Exa")
    return return_results


async def search_with_exa(
    query: str, exa_config: Optional[ExaConfig] = None
) -> UrlsToHtmls:
    """Search the web using Exa and return a dictionary of URLs to content.

    This function performs a web search via Exa's AI-powered search engine and returns
    the results in the same format as fetch_page_content, making it easy to feed
    search results into the cognee pipeline.

    Args:
        query: The search query string.
        exa_config: Configuration for Exa API. If None, defaults are used.

    Returns:
        Dict[str, str]: A dictionary mapping each result URL to its content.

    Raises:
        ImportError: If exa-py is not installed.
        Exception: If the Exa API request fails.
    """
    try:
        from exa_py import Exa
    except ImportError:
        logger.error(
            "Failed to import exa_py, make sure to install using pip install exa-py>=2.0.0"
        )
        raise

    config = exa_config or ExaConfig()

    logger.info(f"Initializing Exa client for web search: '{query}'")
    client = Exa(api_key=config.api_key)
    client.headers["x-exa-integration"] = "cognee"

    search_kwargs: Dict = {
        "query": query,
        "num_results": config.num_results,
        "type": config.search_type,
    }

    # Content retrieval options
    if config.use_text:
        search_kwargs["text"] = {"max_characters": config.text_max_characters}
    if config.use_highlights:
        search_kwargs["highlights"] = {"max_characters": config.highlights_max_characters}
    if config.use_summary:
        search_kwargs["summary"] = True

    # Filtering options
    if config.include_domains:
        search_kwargs["include_domains"] = config.include_domains
    if config.exclude_domains:
        search_kwargs["exclude_domains"] = config.exclude_domains
    if config.include_text:
        search_kwargs["include_text"] = config.include_text
    if config.exclude_text:
        search_kwargs["exclude_text"] = config.exclude_text
    if config.category:
        search_kwargs["category"] = config.category
    if config.start_published_date:
        search_kwargs["start_published_date"] = config.start_published_date
    if config.end_published_date:
        search_kwargs["end_published_date"] = config.end_published_date

    logger.info(
        f"Sending search request to Exa API (type={config.search_type}, "
        f"num_results={config.num_results})"
    )
    response = await asyncio.to_thread(client.search_and_contents, **search_kwargs)

    return_results: Dict[str, str] = {}
    for result in response.results:
        url = getattr(result, "url", "") or ""
        content = _extract_exa_content(result)
        if url:
            return_results[url] = content

    logger.info(
        f"Exa search returned {len(return_results)} result(s) for query: '{query}'"
    )
    return return_results


def _extract_exa_content(result) -> str:
    """Extract content from an Exa search result, cascading through available fields.

    Tries text first, then highlights, then summary. Combines title and URL metadata
    with the best available content.

    Args:
        result: An Exa search result object.

    Returns:
        str: The extracted content string.
    """
    parts = []

    title = getattr(result, "title", None)
    if title:
        parts.append(f"Title: {title}")

    url = getattr(result, "url", None)
    if url:
        parts.append(f"URL: {url}")

    published_date = getattr(result, "published_date", None)
    if published_date:
        parts.append(f"Published: {published_date}")

    # Cascade through content types: text > highlights > summary
    text = getattr(result, "text", None)
    highlights = getattr(result, "highlights", None)
    summary = getattr(result, "summary", None)

    if text:
        parts.append(f"\n{text}")
    elif highlights:
        highlight_text = " ... ".join(highlights) if isinstance(highlights, list) else highlights
        parts.append(f"\n{highlight_text}")
    elif summary:
        parts.append(f"\n{summary}")

    return "\n".join(parts) if parts else ""


async def fetch_with_tavily(urls: Union[str, List[str]]) -> UrlsToHtmls:
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

    tavily_config = TavilyConfig()
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

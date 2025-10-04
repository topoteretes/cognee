from tavily import AsyncTavilyClient
from .bs4_crawler import BeautifulSoupCrawler
import os
from .config import TavilyConfig, SoupCrawlerConfig
from typing import Dict, Any, List, Union, Optional, Literal
from cognee.shared.logging_utils import get_logger
import asyncio

logger = get_logger(__name__)


async def fetch_page_content(
    urls: Union[str, List[str]],
    *,
    preferred_tool: Optional[Literal["tavily", "beautifulsoup"]] = "beautifulsoup",
    tavily_config: Optional[TavilyConfig] = None,
    soup_crawler_config: Optional[SoupCrawlerConfig] = None,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Fetch page content using Tavily API if TAVILY_API_KEY is set,
    otherwise fetch using BeautifulSoupCrawler directly.

    Parameters:
        urls: single URL or list of URLs
        extraction_rules: dict mapping field names -> CSS selector or rule
        use_playwright: whether to render JS (BeautifulSoupCrawler)
        playwright_js_wait: seconds to wait for JS to load
        join_all_matches: join all matching elements per rule
        structured: if True, returns structured dict instead of concatenated string (based on extraction_rules field names)

    Returns:
        Dict mapping URL -> extracted string or structured dict
    """
    if preferred_tool == "tavily":
        if tavily_config.api_key is None:
            raise ValueError("TAVILY_API_KEY must be set in TavilyConfig to use Tavily")
        return await fetch_with_tavily(urls)

    elif preferred_tool == "beautifulsoup":
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error(
                "Failed to import bs4, make sure to install using pip install beautifulsoup4>=4.13.1"
            )
        crawler = BeautifulSoupCrawler()
        extraction_rules = soup_crawler_config.extraction_rules
        if extraction_rules is None:
            raise ValueError("extraction_rules must be provided when not using Tavily")
        try:
            results = await crawler.fetch_with_bs4(
                urls,
                extraction_rules,
                use_playwright=soup_crawler_config.use_playwright,
                playwright_js_wait=soup_crawler_config.playwright_js_wait,
                join_all_matches=soup_crawler_config.join_all_matches,
                structured=soup_crawler_config.structured,
            )
            return results
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")


async def fetch_with_tavily(urls: Union[str, List[str]]) -> Dict[str, str]:
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        logger.error(
            "Failed to import tavily, make sure to install using pip install tavily-python>=0.7.0"
        )
    client = AsyncTavilyClient()
    results = await client.extract(urls)
    for failed_result in results.get("failed_results", []):
        logger.warning(f"Failed to fetch {failed_result}")
    return_results = {}
    for results in results.get("results", []):
        return_results[results["url"]] = results["raw_content"]
    return return_results

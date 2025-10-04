from tavily import AsyncTavilyClient
from .bs4_crawler import BeautifulSoupCrawler
import os
from .config import TavilyConfig, SoupCrawlerConfig
from typing import Dict, Any, List, Union, Optional, Literal
from cognee.shared.logging_utils import get_logger
import asyncio

logger = get_logger(__name__)

try:
    from tavily import AsyncTavilyClient
except ImportError:
    logger.error(
        "Failed to import tavily, make sure to install using pip install tavily-python>=0.7.0"
    )

try:
    from bs4 import BeautifulSoup
except ImportError:
    logger.error(
        "Failed to import bs4, make sure to install using pip install beautifulsoup4>=4.13.1"
    )


async def fetch_page_content(
    urls: Union[str, List[str]],
    *,
    preferred_tool: Optional[Literal["tavily", "beautifulsoup"]] = "beautifulsoup",
    extraction_rules: Optional[Dict[str, Any]] = None,
    tavily_config: Optional[TavilyConfig] = None,
    soup_crawler_config: Optional[SoupCrawlerConfig] = None,
    use_playwright: Optional[bool] = False,
    playwright_js_wait: Optional[float] = 0.8,
    join_all_matches: Optional[bool] = False,
    structured: Optional[bool] = False,
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
    if (os.getenv("TAVILY_API_KEY") or tavily_config.api_key) and preferred_tool == "tavily":
        return await fetch_with_tavily(urls)

    else:
        crawler = BeautifulSoupCrawler()
        extraction_rules = extraction_rules or soup_crawler_config.extraction_rules
        if extraction_rules is None:
            raise ValueError("extraction_rules must be provided when not using Tavily")
        try:
            results = await crawler.fetch_with_bs4(
                urls,
                extraction_rules,
                use_playwright=use_playwright,
                playwright_js_wait=playwright_js_wait,
                join_all_matches=join_all_matches,
                structured=structured,
            )
            return results
        except Exception as e:
            logger.error(f"Error fetching page content: {str(e)}")


async def fetch_with_tavily(urls: Union[str, List[str]]) -> Dict[str, str]:
    client = AsyncTavilyClient()
    results = await client.extract(urls)
    for failed_result in results.get("failed_results", []):
        logger.warning(f"Failed to fetch {failed_result}")
    return_results = {}
    for results in results.get("results", []):
        return_results[results["url"]] = results["raw_content"]
    return return_results


def check_valid_arguments_for_web_scraper(
    extraction_rules, preferred_tool, tavily_config, soup_crawler_config
):
    if preferred_tool == "tavily":
        if not (os.getenv("TAVILY_API_KEY") or (tavily_config and tavily_config.api_key)):
            raise ValueError(
                "TAVILY_API_KEY must be set in environment variables or tavily_config.api_key must be provided when preferred_tool is 'tavily'"
            )
    else:
        print(preferred_tool)
        print(soup_crawler_config)
        print(soup_crawler_config and soup_crawler_config.extraction_rules)
        if not (extraction_rules or (soup_crawler_config and soup_crawler_config.extraction_rules)):
            raise ValueError(
                "extraction_rules must be provided when preferred_tool is 'beautifulsoup'"
            )

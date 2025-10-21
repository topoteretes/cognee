import os
import pytest
from cognee.tasks.web_scraper.config import TavilyConfig
from cognee.tasks.web_scraper.utils import fetch_with_tavily


@pytest.mark.asyncio
async def test_fetch():
    url = "https://en.wikipedia.org/wiki/Large_language_model"
    tavily_config = TavilyConfig()
    results = await fetch_with_tavily(url, tavily_config)
    assert len(results) == 1
    assert isinstance(results, dict)
    html = results[url]
    assert isinstance(html, str)

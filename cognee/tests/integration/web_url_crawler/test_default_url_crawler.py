import pytest
from cognee.tasks.web_scraper import DefaultUrlCrawler


@pytest.mark.asyncio
async def test_fetch():
    crawler = DefaultUrlCrawler()
    url = "https://en.wikipedia.org/wiki/Large_language_model"
    results = await crawler.fetch_urls(url)
    assert len(results) == 1
    assert isinstance(results, dict)
    html = results[url]
    assert isinstance(html, str)

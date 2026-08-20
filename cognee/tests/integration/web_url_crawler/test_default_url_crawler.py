import pytest
from cognee.tasks.web_scraper import DefaultUrlCrawler


@pytest.mark.asyncio
async def test_fetch():
    crawler = DefaultUrlCrawler()
    url = "http://example.com/"
    results = await crawler.fetch_urls(url)
    assert len(results) == 1
    assert isinstance(results, dict)
    html = results[url]
    assert isinstance(html, str)


@pytest.mark.asyncio
async def test_fetch_accepts_list():
    crawler = DefaultUrlCrawler()
    urls = ["http://example.com/"]
    results = await crawler.fetch_urls(urls)
    assert isinstance(results, dict)
    assert len(results) == 1
    html = results["http://example.com/"]
    assert isinstance(html, str)

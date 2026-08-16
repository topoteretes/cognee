import asyncio

import pytest

from cognee.tasks.web_scraper import DefaultUrlCrawler


@pytest.mark.asyncio
async def test_rate_limit_serializes_concurrent_requests_to_same_domain():
    crawler = DefaultUrlCrawler(crawl_delay=0.05)
    request_times = []

    async def request(url):
        await crawler._respect_rate_limit(url)
        request_times.append(asyncio.get_running_loop().time())

    await asyncio.gather(
        request("https://example.test/one"),
        request("https://example.test/two"),
        request("https://example.test/three"),
    )

    request_times.sort()
    gaps = [later - earlier for earlier, later in zip(request_times, request_times[1:])]
    assert all(gap >= 0.04 for gap in gaps), gaps

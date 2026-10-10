import asyncio

import pytest

from cognee.tasks.web_scraper import default_url_crawler as crawler_module
from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler


@pytest.mark.asyncio
async def test_rate_limit_serializes_concurrent_requests_for_same_domain(monkeypatch):
    current_time = 0.0
    real_sleep = asyncio.sleep

    async def advance_clock(delay: float):
        nonlocal current_time
        wake_time = current_time + delay
        await real_sleep(0)
        current_time = max(current_time, wake_time)

    monkeypatch.setattr(crawler_module.time, "time", lambda: current_time)
    monkeypatch.setattr(crawler_module.asyncio, "sleep", advance_clock)

    crawler = DefaultUrlCrawler(crawl_delay=1.0)
    request_times = []

    async def make_request(path: str):
        await crawler._respect_rate_limit(f"https://example.test/{path}")
        request_times.append(current_time)

    await asyncio.gather(*(make_request(path) for path in ("a", "b", "c")))

    assert sorted(request_times) == [0.0, 1.0, 2.0]

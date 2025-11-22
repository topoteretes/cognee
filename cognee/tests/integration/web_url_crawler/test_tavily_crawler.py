import os
import pytest
from cognee.tasks.web_scraper.utils import fetch_with_tavily

skip_in_ci = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping in Github for now - before we get TAVILY_API_KEY",
)


@skip_in_ci
@pytest.mark.asyncio
async def test_fetch():
    url = "http://example.com/"
    results = await fetch_with_tavily(url)
    assert isinstance(results, dict)
    assert len(results) == 1
    html = results[url]
    assert isinstance(html, str)

import os
import pytest
from cognee.tasks.web_scraper.utils import fetch_with_keenable

skip_in_ci = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping in Github for now - before we get KEENABLE_API_KEY",
)

skip_without_key = pytest.mark.skipif(
    not os.getenv("KEENABLE_API_KEY"),
    reason="KEENABLE_API_KEY is not set",
)


@skip_in_ci
@skip_without_key
@pytest.mark.asyncio
async def test_fetch():
    url = "http://example.com/"
    results = await fetch_with_keenable(url)
    assert isinstance(results, dict)
    assert len(results) == 1
    content = results[url]
    assert isinstance(content, str)

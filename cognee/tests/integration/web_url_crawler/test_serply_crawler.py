import os

import pytest

from cognee.tasks.web_scraper.utils import fetch_with_serply

skip_in_ci = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping in Github for now - before we get SERPLY_API_KEY",
)

skip_without_key = pytest.mark.skipif(
    not os.getenv("SERPLY_API_KEY"),
    reason="SERPLY_API_KEY is not set",
)


@skip_in_ci
@skip_without_key
@pytest.mark.asyncio
async def test_fetch():
    url = "http://example.com/"
    results = await fetch_with_serply(url)
    assert isinstance(results, dict)
    assert len(results) == 1
    content = results[url]
    assert isinstance(content, str)

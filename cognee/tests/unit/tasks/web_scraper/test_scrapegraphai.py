"""Unit tests for ScrapeGraphAI integration in fetch_with_scrapegraphai."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_client(markdownify_side_effect):
    """Helper to build a mock AsyncClient context manager."""
    mock_client = AsyncMock()
    if isinstance(markdownify_side_effect, list):
        mock_client.markdownify = AsyncMock(side_effect=markdownify_side_effect)
    else:
        mock_client.markdownify = AsyncMock(return_value=markdownify_side_effect)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


@pytest.mark.asyncio
async def test_fetch_with_scrapegraphai_single_url():
    """Test that fetch_with_scrapegraphai fetches a single URL and returns markdown content."""
    mock_response = {
        "request_id": "mock-req-123",
        "status": "completed",
        "result": "# Test Page\n\nThis is test markdown content.",
    }
    mock_client = _make_mock_client(mock_response)

    with patch.dict("os.environ", {"SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000"}):
        with patch("cognee.tasks.web_scraper.utils.ScrapeGraphAIConfig") as mock_config_cls:
            mock_config_cls.return_value.api_key = "sgai-test-00000000-0000-0000-0000-000000000000"
            mock_config_cls.return_value.stealth = False
            mock_config_cls.return_value.render_heavy_js = False

            with patch("scrapegraph_py.AsyncClient", MagicMock(return_value=mock_client)):
                from cognee.tasks.web_scraper.utils import fetch_with_scrapegraphai

                result = await fetch_with_scrapegraphai("https://example.com/")

    assert "https://example.com/" in result
    assert result["https://example.com/"] == "# Test Page\n\nThis is test markdown content."
    mock_client.markdownify.assert_awaited_once_with(
        website_url="https://example.com/",
        stealth=False,
        render_heavy_js=False,
    )


@pytest.mark.asyncio
async def test_fetch_with_scrapegraphai_multiple_urls():
    """Test that fetch_with_scrapegraphai handles multiple URLs concurrently."""
    urls = ["https://example.com/page1", "https://example.com/page2"]
    mock_client = _make_mock_client(
        [
            {"request_id": "mock-req-1", "status": "completed", "result": "# Page 1"},
            {"request_id": "mock-req-2", "status": "completed", "result": "# Page 2"},
        ]
    )

    with patch.dict("os.environ", {"SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000"}):
        with patch("cognee.tasks.web_scraper.utils.ScrapeGraphAIConfig") as mock_config_cls:
            mock_config_cls.return_value.api_key = "sgai-test-00000000-0000-0000-0000-000000000000"
            mock_config_cls.return_value.stealth = False
            mock_config_cls.return_value.render_heavy_js = False

            with patch("scrapegraph_py.AsyncClient", MagicMock(return_value=mock_client)):
                from cognee.tasks.web_scraper.utils import fetch_with_scrapegraphai

                result = await fetch_with_scrapegraphai(urls)

    assert len(result) == 2
    assert result["https://example.com/page1"] == "# Page 1"
    assert result["https://example.com/page2"] == "# Page 2"


@pytest.mark.asyncio
async def test_fetch_with_scrapegraphai_skips_failed_urls():
    """Test that fetch_with_scrapegraphai skips URLs where the API raises an exception."""
    urls = ["https://good.example.com/", "https://bad.example.com/"]
    mock_client = _make_mock_client(
        [
            {"request_id": "mock-req-1", "status": "completed", "result": "# Good Page"},
            Exception("API error: rate limit exceeded"),
        ]
    )

    with patch.dict("os.environ", {"SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000"}):
        with patch("cognee.tasks.web_scraper.utils.ScrapeGraphAIConfig") as mock_config_cls:
            mock_config_cls.return_value.api_key = "sgai-test-00000000-0000-0000-0000-000000000000"
            mock_config_cls.return_value.stealth = False
            mock_config_cls.return_value.render_heavy_js = False

            with patch("scrapegraph_py.AsyncClient", MagicMock(return_value=mock_client)):
                from cognee.tasks.web_scraper.utils import fetch_with_scrapegraphai

                result = await fetch_with_scrapegraphai(urls)

    assert "https://good.example.com/" in result
    assert "https://bad.example.com/" not in result
    assert result["https://good.example.com/"] == "# Good Page"


@pytest.mark.asyncio
async def test_fetch_with_scrapegraphai_skips_empty_results():
    """Test that fetch_with_scrapegraphai skips URLs with empty result fields."""
    mock_client = _make_mock_client(
        {"request_id": "mock-req-123", "status": "completed", "result": ""}
    )

    with patch.dict("os.environ", {"SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000"}):
        with patch("cognee.tasks.web_scraper.utils.ScrapeGraphAIConfig") as mock_config_cls:
            mock_config_cls.return_value.api_key = "sgai-test-00000000-0000-0000-0000-000000000000"
            mock_config_cls.return_value.stealth = False
            mock_config_cls.return_value.render_heavy_js = False

            with patch("scrapegraph_py.AsyncClient", MagicMock(return_value=mock_client)):
                from cognee.tasks.web_scraper.utils import fetch_with_scrapegraphai

                result = await fetch_with_scrapegraphai("https://example.com/")

    assert len(result) == 0


@pytest.mark.asyncio
async def test_fetch_page_content_prefers_scrapegraphai_over_tavily():
    """Test that fetch_page_content uses ScrapeGraphAI when SGAI_API_KEY is set."""
    with patch.dict(
        "os.environ",
        {
            "SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000",
            "TAVILY_API_KEY": "tavily-test-key",
        },
    ):
        with patch(
            "cognee.tasks.web_scraper.utils.fetch_with_scrapegraphai",
            new_callable=AsyncMock,
            return_value={"https://example.com/": "# Scraped via ScrapeGraphAI"},
        ) as mock_sgai:
            with patch(
                "cognee.tasks.web_scraper.utils.fetch_with_tavily",
                new_callable=AsyncMock,
            ) as mock_tavily:
                from cognee.tasks.web_scraper.utils import fetch_page_content

                result = await fetch_page_content("https://example.com/")

        mock_sgai.assert_awaited_once()
        mock_tavily.assert_not_awaited()
        assert result == {"https://example.com/": "# Scraped via ScrapeGraphAI"}


def test_scrapegraphai_config_defaults():
    """Test ScrapeGraphAIConfig defaults and env var loading."""
    import os
    from unittest.mock import patch

    with patch.dict("os.environ", {"SGAI_API_KEY": "sgai-test-00000000-0000-0000-0000-000000000000"}, clear=False):
        from cognee.tasks.web_scraper.config import ScrapeGraphAIConfig

        config = ScrapeGraphAIConfig()
        assert config.api_key == "sgai-test-00000000-0000-0000-0000-000000000000"
        assert config.stealth is False
        assert config.render_heavy_js is False


def test_scrapegraphai_config_custom_values():
    """Test ScrapeGraphAIConfig with custom values."""
    from cognee.tasks.web_scraper.config import ScrapeGraphAIConfig

    config = ScrapeGraphAIConfig(
        api_key="sgai-custom-00000000-0000-0000-0000-000000000000",
        stealth=True,
        render_heavy_js=True,
    )
    assert config.api_key == "sgai-custom-00000000-0000-0000-0000-000000000000"
    assert config.stealth is True
    assert config.render_heavy_js is True

import os
from unittest.mock import MagicMock, patch

import pytest

from cognee.tasks.web_scraper.config import ExaConfig
from cognee.tasks.web_scraper.utils import _extract_exa_content


# ---------------------------------------------------------------------------
# Fixtures: mock Exa result objects
# ---------------------------------------------------------------------------


def _make_result(title="Example", url="https://example.com", text=None,
                 highlights=None, summary=None, published_date=None):
    """Create a mock Exa search result with the given fields."""
    result = MagicMock()
    result.title = title
    result.url = url
    result.text = text
    result.highlights = highlights
    result.summary = summary
    result.published_date = published_date
    return result


# ---------------------------------------------------------------------------
# ExaConfig tests
# ---------------------------------------------------------------------------


class TestExaConfig:
    def test_defaults(self):
        config = ExaConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.num_results == 10
        assert config.search_type == "auto"
        assert config.use_highlights is True
        assert config.use_text is True
        assert config.use_summary is False
        assert config.include_domains is None
        assert config.category is None

    def test_custom_values(self):
        config = ExaConfig(
            api_key="test-key",
            num_results=5,
            search_type="neural",
            category="news",
            include_domains=["example.com"],
            start_published_date="2024-01-01T00:00:00Z",
        )
        assert config.num_results == 5
        assert config.search_type == "neural"
        assert config.category == "news"
        assert config.include_domains == ["example.com"]
        assert config.start_published_date == "2024-01-01T00:00:00Z"

    def test_api_key_passed_explicitly(self):
        """API key can be set directly (env var default is evaluated at class load time)."""
        config = ExaConfig(api_key="explicit-key")
        assert config.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Content extraction tests
# ---------------------------------------------------------------------------


class TestExtractExaContent:
    def test_text_content(self):
        result = _make_result(text="Full page text content here.")
        content = _extract_exa_content(result)
        assert "Title: Example" in content
        assert "URL: https://example.com" in content
        assert "Full page text content here." in content

    def test_highlights_fallback(self):
        """When text is absent, highlights should be used."""
        result = _make_result(
            text=None,
            highlights=["First highlight.", "Second highlight."],
        )
        content = _extract_exa_content(result)
        assert "First highlight." in content
        assert "Second highlight." in content
        assert " ... " in content  # joined with separator

    def test_summary_fallback(self):
        """When both text and highlights are absent, summary should be used."""
        result = _make_result(text=None, highlights=None, summary="A page summary.")
        content = _extract_exa_content(result)
        assert "A page summary." in content

    def test_no_content(self):
        """When all content fields are absent, still returns title/URL metadata."""
        result = _make_result(text=None, highlights=None, summary=None)
        content = _extract_exa_content(result)
        assert "Title: Example" in content
        assert "URL: https://example.com" in content

    def test_published_date(self):
        result = _make_result(
            text="Some text", published_date="2024-06-15T00:00:00Z"
        )
        content = _extract_exa_content(result)
        assert "Published: 2024-06-15" in content

    def test_empty_result(self):
        """A result with no fields should return empty string."""
        result = _make_result(
            title=None, url=None, text=None, highlights=None, summary=None
        )
        content = _extract_exa_content(result)
        assert content == ""


# ---------------------------------------------------------------------------
# fetch_with_exa tests
# ---------------------------------------------------------------------------


class TestFetchWithExa:
    @pytest.mark.asyncio
    async def test_fetch_with_exa_parses_response(self):
        mock_result = _make_result(
            url="https://example.com",
            text="Page content from Exa.",
        )
        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_exa_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.get_contents.return_value = mock_response
        mock_exa_cls.return_value = mock_client

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=mock_exa_cls)}):
            from cognee.tasks.web_scraper.utils import fetch_with_exa

            config = ExaConfig(api_key="test-key")
            results = await fetch_with_exa("https://example.com", exa_config=config)

        assert isinstance(results, dict)
        assert "https://example.com" in results
        assert "Page content from Exa." in results["https://example.com"]
        assert mock_client.headers["x-exa-integration"] == "cognee"

    @pytest.mark.asyncio
    async def test_fetch_with_exa_multiple_urls(self):
        mock_results = [
            _make_result(url="https://a.com", text="Content A"),
            _make_result(url="https://b.com", text="Content B"),
        ]
        mock_response = MagicMock()
        mock_response.results = mock_results

        mock_exa_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.get_contents.return_value = mock_response
        mock_exa_cls.return_value = mock_client

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=mock_exa_cls)}):
            from cognee.tasks.web_scraper.utils import fetch_with_exa

            config = ExaConfig(api_key="test-key")
            results = await fetch_with_exa(
                ["https://a.com", "https://b.com"], exa_config=config
            )

        assert len(results) == 2
        assert "Content A" in results["https://a.com"]
        assert "Content B" in results["https://b.com"]


# ---------------------------------------------------------------------------
# search_with_exa tests
# ---------------------------------------------------------------------------


class TestSearchWithExa:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_results = [
            _make_result(
                title="AI Research",
                url="https://ai.example.com",
                text="New developments in AI.",
            ),
            _make_result(
                title="ML Guide",
                url="https://ml.example.com",
                highlights=["Machine learning basics."],
            ),
        ]
        mock_response = MagicMock()
        mock_response.results = mock_results

        mock_exa_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.return_value = mock_response
        mock_exa_cls.return_value = mock_client

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=mock_exa_cls)}):
            from cognee.tasks.web_scraper.utils import search_with_exa

            config = ExaConfig(api_key="test-key", num_results=5, search_type="neural")
            results = await search_with_exa("AI research", exa_config=config)

        assert len(results) == 2
        assert "https://ai.example.com" in results
        assert "New developments in AI." in results["https://ai.example.com"]
        assert "Machine learning basics." in results["https://ml.example.com"]
        assert mock_client.headers["x-exa-integration"] == "cognee"

        # Verify search parameters were passed correctly
        call_kwargs = mock_client.search_and_contents.call_args
        assert call_kwargs.kwargs["query"] == "AI research"
        assert call_kwargs.kwargs["num_results"] == 5
        assert call_kwargs.kwargs["type"] == "neural"

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        mock_response = MagicMock()
        mock_response.results = []

        mock_exa_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.search_and_contents.return_value = mock_response
        mock_exa_cls.return_value = mock_client

        with patch.dict("sys.modules", {"exa_py": MagicMock(Exa=mock_exa_cls)}):
            from cognee.tasks.web_scraper.utils import search_with_exa

            config = ExaConfig(
                api_key="test-key",
                category="news",
                include_domains=["reuters.com", "bbc.com"],
                exclude_text=["sponsored"],
                start_published_date="2024-01-01T00:00:00Z",
            )
            await search_with_exa("latest news", exa_config=config)

        call_kwargs = mock_client.search_and_contents.call_args.kwargs
        assert call_kwargs["category"] == "news"
        assert call_kwargs["include_domains"] == ["reuters.com", "bbc.com"]
        assert call_kwargs["exclude_text"] == ["sponsored"]
        assert call_kwargs["start_published_date"] == "2024-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Disabled state: no EXA_API_KEY
# ---------------------------------------------------------------------------


class TestExaDisabledState:
    def test_config_has_no_key_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("EXA_API_KEY", None)
            config = ExaConfig()
            assert config.api_key is None

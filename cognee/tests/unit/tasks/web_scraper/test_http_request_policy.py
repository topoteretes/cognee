"""ALLOW_HTTP_REQUESTS is opt-in on every surface, and refusal precedes any outbound work."""

import importlib
from unittest.mock import AsyncMock, Mock

import pytest

from cognee.modules.operations.origin import operation_origin_scope
from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler
from cognee.tasks.web_scraper.ssrf_protection import (
    SSRFProtectionError,
    is_http_requests_allowed,
    validate_outbound_url,
)

ssrf = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")
fetching = importlib.import_module("cognee.tasks.web_scraper.utils")

UNSET_OR_BLANK = [None, "", "   "]


def _configure(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)


@pytest.mark.parametrize("setting", UNSET_OR_BLANK)
def test_unset_and_blank_values_refuse(monkeypatch, setting):
    _configure(monkeypatch, setting)
    assert is_http_requests_allowed() is False


@pytest.mark.parametrize(
    "setting,expected",
    [
        ("true", True),
        ("True", True),
        (" true ", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_explicit_values(monkeypatch, setting, expected):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)
    assert is_http_requests_allowed() is expected


@pytest.mark.parametrize("origin", ["api", "sdk", "cli", "mcp", "background"])
def test_the_rule_does_not_depend_on_the_call_origin(monkeypatch, origin):
    """One rule everywhere: a missing origin stamp can never fail open."""
    with operation_origin_scope(origin):
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
        assert is_http_requests_allowed() is False
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
        assert is_http_requests_allowed() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", UNSET_OR_BLANK + ["false"])
async def test_refusal_names_the_opt_in_before_dns(monkeypatch, setting):
    _configure(monkeypatch, setting)
    resolve = AsyncMock()
    monkeypatch.setattr(ssrf, "_resolve_host_ips", resolve)

    with pytest.raises(SSRFProtectionError, match="Set ALLOW_HTTP_REQUESTS=true") as error:
        await validate_outbound_url("https://example.com/document")

    assert error.value.status_code == 403
    assert "on the server" in error.value.message
    if setting == "false":
        assert "ALLOW_HTTP_REQUESTS=false" in error.value.message
    else:
        assert "ALLOW_HTTP_REQUESTS is unset" in error.value.message
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_opt_in_still_applies_ssrf_checks(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    await validate_outbound_url("https://8.8.8.8/document")
    with pytest.raises(SSRFProtectionError, match="non-public"):
        await validate_outbound_url("http://127.0.0.1/private")


@pytest.mark.asyncio
async def test_disabled_crawler_raises_instead_of_returning_empty_content(monkeypatch):
    monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    crawler = DefaultUrlCrawler()
    check_robots = AsyncMock(return_value=True)
    fetch = AsyncMock(return_value="<html>content</html>")
    monkeypatch.setattr(crawler, "_is_url_allowed", check_robots)
    monkeypatch.setattr(crawler, "_fetch_httpx", fetch)

    with pytest.raises(SSRFProtectionError, match="ALLOW_HTTP_REQUESTS=true"):
        await crawler.fetch_urls("https://8.8.8.8/document")

    check_robots.assert_not_awaited()
    fetch.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["beautifulsoup", "tavily", "keenable"])
async def test_disabled_fetching_never_initializes_a_scraping_backend(monkeypatch, backend):
    monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    tavily = AsyncMock()
    keenable = AsyncMock()
    crawler = Mock()
    monkeypatch.setattr(fetching, "fetch_with_tavily", tavily)
    monkeypatch.setattr(fetching, "fetch_with_keenable", keenable)
    monkeypatch.setattr(fetching, "DefaultUrlCrawler", crawler)

    with pytest.raises(SSRFProtectionError, match="ALLOW_HTTP_REQUESTS=true"):
        await fetching.fetch_page_content("https://example.com/document", preferred_tool=backend)

    tavily.assert_not_awaited()
    keenable.assert_not_awaited()
    crawler.assert_not_called()

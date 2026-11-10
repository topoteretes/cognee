"""API defaults, explicit overrides, and refusal before any outbound work."""

import asyncio
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


@pytest.mark.parametrize(
    "origin,expected",
    [("api", False), ("sdk", True), ("cli", True), ("mcp", True), ("background", True)],
)
def test_unset_default_depends_on_the_call_origin(monkeypatch, origin, expected):
    monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    with operation_origin_scope(origin):
        assert is_http_requests_allowed() is expected


@pytest.mark.parametrize("origin,expected", [("api", False), ("sdk", True)])
@pytest.mark.parametrize("setting", ["", "   "])
def test_blank_value_counts_as_unset(monkeypatch, origin, expected, setting):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)
    with operation_origin_scope(origin):
        assert is_http_requests_allowed() is expected


@pytest.mark.parametrize("origin", ["api", "sdk", "cli", "mcp", "background"])
@pytest.mark.parametrize(
    "setting,expected",
    [
        ("true", True),
        (" TRUE ", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("off", False),
        # Only recognised true values enable fetching: a typo fails closed.
        ("flase", False),
        ("ture", False),
        ("disabled", False),
        ("none", False),
    ],
)
def test_explicit_setting_overrides_the_origin(monkeypatch, origin, setting, expected):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)
    with operation_origin_scope(origin):
        assert is_http_requests_allowed() is expected


@pytest.mark.asyncio
async def test_api_default_is_inherited_by_child_tasks_without_changing_sdk_calls(monkeypatch):
    monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)

    async def allowed():
        await asyncio.sleep(0)
        return is_http_requests_allowed()

    with operation_origin_scope("api"):
        api_task = asyncio.create_task(allowed())
    assert await asyncio.gather(api_task, allowed()) == [False, True]
    assert "ALLOW_HTTP_REQUESTS" not in ssrf.os.environ


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, "", "false", "flase"])
async def test_api_refusal_names_the_opt_in_before_dns(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", setting)
    resolve = AsyncMock()
    monkeypatch.setattr(ssrf, "_resolve_host_ips", resolve)

    with (
        operation_origin_scope("api"),
        pytest.raises(SSRFProtectionError, match="Set ALLOW_HTTP_REQUESTS=true") as error,
    ):
        await validate_outbound_url("https://example.com/document")

    assert error.value.status_code == 403
    assert "on the server" in error.value.message
    if setting:
        # The configured value is echoed, with the values that would enable fetching.
        assert f"ALLOW_HTTP_REQUESTS={setting};" in error.value.message
        assert "only true, 1, yes or on" in error.value.message
    else:
        assert "ALLOW_HTTP_REQUESTS is unset" in error.value.message
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_opt_in_still_applies_ssrf_checks(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    with operation_origin_scope("api"):
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

    with (
        operation_origin_scope("api"),
        pytest.raises(SSRFProtectionError, match="ALLOW_HTTP_REQUESTS=true"),
    ):
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

    with (
        operation_origin_scope("api"),
        pytest.raises(SSRFProtectionError, match="ALLOW_HTTP_REQUESTS=true"),
    ):
        await fetching.fetch_page_content("https://example.com/document", preferred_tool=backend)

    tavily.assert_not_awaited()
    keenable.assert_not_awaited()
    crawler.assert_not_called()

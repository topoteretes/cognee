"""Unit tests for the SSRF protection guard used by outbound URL fetching."""

import ipaddress

import pytest

from cognee.tasks.web_scraper import ssrf_protection as ssrf
from cognee.tasks.web_scraper.ssrf_protection import (
    SSRFProtectionError,
    is_http_requests_allowed,
    validate_outbound_url,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:6379/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (IMDS)
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",  # IPv6 loopback
    ],
)
async def test_blocks_internal_ip_literals(url, monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    with pytest.raises(SSRFProtectionError):
        await validate_outbound_url(url)


@pytest.mark.asyncio
async def test_allows_public_ip_literal(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    # A public IP literal needs no DNS lookup and must be permitted.
    await validate_outbound_url("http://8.8.8.8/")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/resource",
        "gopher://example.com/",
        "//example.com/no-scheme",
    ],
)
async def test_blocks_disallowed_schemes(url, monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    with pytest.raises(SSRFProtectionError):
        await validate_outbound_url(url)


@pytest.mark.asyncio
async def test_blocks_url_without_host(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    with pytest.raises(SSRFProtectionError):
        await validate_outbound_url("http:///path-only")


@pytest.mark.asyncio
async def test_disabled_outbound_http_blocks_even_public(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")
    with pytest.raises(SSRFProtectionError, match="disabled"):
        await validate_outbound_url("http://8.8.8.8/")


@pytest.mark.asyncio
async def test_hostname_resolving_to_internal_is_blocked(monkeypatch):
    """A public-looking hostname that resolves to an internal IP must be blocked
    (defends against DNS rebinding / internal DNS names)."""
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")

    async def fake_resolve(host):
        return [ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(ssrf, "_resolve_host_ips", fake_resolve)

    with pytest.raises(SSRFProtectionError, match="non-public"):
        await validate_outbound_url("http://evil.example.com/")


@pytest.mark.asyncio
async def test_hostname_resolving_to_public_is_allowed(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")

    async def fake_resolve(host):
        return [ipaddress.ip_address("93.184.216.34")]  # example.com

    monkeypatch.setattr(ssrf, "_resolve_host_ips", fake_resolve)

    await validate_outbound_url("http://example.com/page")


@pytest.mark.asyncio
async def test_unresolvable_host_is_blocked(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")

    async def fake_resolve(host):
        return []

    monkeypatch.setattr(ssrf, "_resolve_host_ips", fake_resolve)

    with pytest.raises(SSRFProtectionError, match="resolve"):
        await validate_outbound_url("http://nonexistent.invalid/")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),  # unset -> default enabled
        ("true", True),
        ("True", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_is_http_requests_allowed_parsing(value, expected, monkeypatch):
    if value is None:
        monkeypatch.delenv("ALLOW_HTTP_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("ALLOW_HTTP_REQUESTS", value)
    assert is_http_requests_allowed() is expected

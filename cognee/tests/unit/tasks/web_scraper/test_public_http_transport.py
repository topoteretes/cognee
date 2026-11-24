"""Drive HTTPX/httpcore through DNS and TLS to test the socket boundary."""

import ipaddress
from unittest.mock import AsyncMock

import httpcore
import httpx
import pytest

from cognee.tasks.web_scraper import ssrf_protection as ssrf
from cognee.tasks.web_scraper.public_http_transport import PublicHTTPTransport


class Stream(httpcore.AsyncNetworkStream):
    def __init__(self, response, records):
        self.response = response
        self.records = records

    async def read(self, max_bytes, timeout=None):
        data, self.response = self.response, b""
        return data

    async def write(self, buffer, timeout=None):
        self.records.append(("write", buffer))

    async def aclose(self):
        pass

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        assert ssl_context.check_hostname
        self.records.append(("sni", server_hostname))
        return self

    def get_extra_info(self, info):
        return None


@pytest.fixture
def client_parts(monkeypatch):
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    monkeypatch.setenv("HTTPS_PROXY", "http://169.254.169.254:80")
    transport = PublicHTTPTransport()
    records = []
    responses = [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]

    async def connect(host, port, **kwargs):
        records.append(("connect", host, port))
        return Stream(responses.pop(0), records)

    monkeypatch.setattr(transport._pool._network_backend.backend, "connect_tcp", connect)
    return transport, records, responses


@pytest.mark.asyncio
async def test_dns_is_pinned_while_host_and_sni_keep_original_name(monkeypatch, client_parts):
    transport, records, _ = client_parts
    resolve = AsyncMock(
        side_effect=[[ipaddress.ip_address("93.184.216.34")], [ipaddress.ip_address("127.0.0.1")]]
    )
    monkeypatch.setattr(ssrf, "_resolve_host_ips", resolve)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        assert (await client.get("https://public.example/page")).text == "ok"
    assert resolve.await_count == 1
    assert ("connect", "93.184.216.34", 443) in records
    assert ("sni", "public.example") in records
    assert any(r[0] == "write" and b"Host: public.example\r\n" in r[1] for r in records)


@pytest.mark.asyncio
async def test_rebind_between_preflight_and_fetch_never_opens_socket(monkeypatch, client_parts):
    transport, records, _ = client_parts
    monkeypatch.setattr(
        ssrf,
        "_resolve_host_ips",
        AsyncMock(
            side_effect=[
                [ipaddress.ip_address("93.184.216.34")],
                [ipaddress.ip_address("10.0.0.1")],
            ]
        ),
    )
    await ssrf.validate_outbound_url("https://public.example/")
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(ssrf.SSRFProtectionError):
            await client.get("https://public.example/")
    assert not records


@pytest.mark.asyncio
async def test_redirect_cannot_reach_metadata(monkeypatch, client_parts):
    transport, records, responses = client_parts
    responses[:] = [
        b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\nContent-Length: 0\r\n\r\n"
    ]
    monkeypatch.setattr(
        ssrf, "_resolve_host_ips", AsyncMock(return_value=[ipaddress.ip_address("93.184.216.34")])
    )
    async with httpx.AsyncClient(
        transport=transport, trust_env=False, follow_redirects=True
    ) as client:
        with pytest.raises(ssrf.SSRFProtectionError):
            await client.get("https://public.example/")
    assert [r for r in records if r[0] == "connect"] == [("connect", "93.184.216.34", 443)]


@pytest.mark.asyncio
async def test_shared_ip_does_not_share_tls_identity(monkeypatch, client_parts):
    transport, records, responses = client_parts
    responses.append(responses[0])
    monkeypatch.setattr(
        ssrf, "_resolve_host_ips", AsyncMock(return_value=[ipaddress.ip_address("93.184.216.34")])
    )
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        await client.get("https://first.example/")
        await client.get("https://second.example/")
    assert [r for r in records if r[0] == "sni"] == [
        ("sni", "first.example"),
        ("sni", "second.example"),
    ]


@pytest.mark.asyncio
async def test_mixed_dns_answer_is_rejected_before_connect(monkeypatch, client_parts):
    transport, records, _ = client_parts
    monkeypatch.setattr(
        ssrf,
        "_resolve_host_ips",
        AsyncMock(
            return_value=[ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("100.64.0.1")]
        ),
    )
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        with pytest.raises(ssrf.SSRFProtectionError):
            await client.get("https://mixed.example/")
    assert not records


@pytest.mark.asyncio
async def test_browser_subresource_uses_same_boundary(monkeypatch, client_parts):
    from types import SimpleNamespace

    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    transport, records, _ = client_parts
    route = SimpleNamespace(
        request=SimpleNamespace(
            method="GET",
            url="http://169.254.169.254/",
            all_headers=AsyncMock(return_value={}),
            post_data_buffer=None,
        ),
        abort=AsyncMock(),
        fulfill=AsyncMock(),
    )
    crawler = DefaultUrlCrawler()
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        crawler._client = client
        with pytest.raises(ssrf.SSRFProtectionError):
            await crawler._fetch_browser_route(route)
    route.abort.assert_awaited_once_with("blockedbyclient")
    route.fulfill.assert_not_called()
    assert not records

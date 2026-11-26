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
            resource_type="fetch",
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


@pytest.mark.asyncio
async def test_production_client_is_pinned_and_does_not_mount_env_proxies(monkeypatch):
    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    async with DefaultUrlCrawler() as crawler:
        assert isinstance(crawler._client._transport, PublicHTTPTransport)
        assert not crawler._client._mounts


@pytest.mark.asyncio
async def test_browser_routes_are_registered_before_page_and_native_network_is_disabled(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from cognee.tasks.web_scraper import default_url_crawler as module

    events = []
    page = SimpleNamespace(goto=AsyncMock(), content=AsyncMock(return_value="<html>ok</html>"))
    context = SimpleNamespace(
        route=AsyncMock(side_effect=lambda *a: events.append("http")),
        route_web_socket=AsyncMock(side_effect=lambda *a: events.append("ws")),
        new_page=AsyncMock(side_effect=lambda: events.append("page") or page),
    )
    browser = SimpleNamespace(new_context=AsyncMock(return_value=context), close=AsyncMock())
    launch = AsyncMock(return_value=browser)
    playwright = MagicMock()
    playwright.__aenter__ = AsyncMock(
        return_value=SimpleNamespace(chromium=SimpleNamespace(launch=launch))
    )
    playwright.__aexit__ = AsyncMock()
    monkeypatch.setattr(module, "async_playwright", lambda: playwright)
    async with module.DefaultUrlCrawler(max_retries=0) as crawler:
        assert (
            await crawler._render_with_playwright("https://public.example", js_wait=0)
            == "<html>ok</html>"
        )
    socket = SimpleNamespace(close=AsyncMock())
    await context.route_web_socket.call_args.args[1](socket)
    socket.close.assert_awaited_once()
    assert events == ["http", "ws", "page"]
    assert launch.call_args.kwargs["proxy"] == {"server": "http://127.0.0.1:9"}
    assert (
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"
        in launch.call_args.kwargs["args"]
    )
    assert "--host-resolver-rules=MAP * ~NOTFOUND" in launch.call_args.kwargs["args"]
    browser.new_context.assert_awaited_once_with(service_workers="block")


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", ["image", "media", "font"])
async def test_large_assets_are_aborted_without_fetch(resource_type):
    from types import SimpleNamespace

    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    route = SimpleNamespace(request=SimpleNamespace(resource_type=resource_type), abort=AsyncMock())
    await DefaultUrlCrawler()._fetch_browser_route(route)
    route.abort.assert_awaited_once_with("blockedbyclient")


@pytest.mark.asyncio
async def test_streamed_browser_response_exceeding_explicit_limit_is_aborted():
    from types import SimpleNamespace

    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    route = SimpleNamespace(
        request=SimpleNamespace(
            resource_type="fetch",
            method="GET",
            url="https://public.example/large",
            all_headers=AsyncMock(return_value={}),
            post_data_buffer=None,
        ),
        abort=AsyncMock(),
        fulfill=AsyncMock(),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"123456789"))
    ) as client:
        crawler = DefaultUrlCrawler(max_browser_response_bytes=8)
        crawler._client = client
        with pytest.raises(ValueError, match="exceeds"):
            await crawler._fetch_browser_route(route)
    route.abort.assert_awaited_once()
    route.fulfill.assert_not_called()


@pytest.mark.asyncio
async def test_dns_resolution_obeys_connect_timeout(monkeypatch):
    import asyncio

    from cognee.tasks.web_scraper import public_http_transport as module

    async def never(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "validate_outbound_url", never)
    backend = module.PublicNetworkBackend()
    with pytest.raises(httpcore.ConnectTimeout):
        await backend.connect_tcp("public.example", 443, timeout=0.01)


@pytest.mark.asyncio
async def test_addresses_share_one_connect_budget(monkeypatch):
    import asyncio

    from cognee.tasks.web_scraper import public_http_transport as module

    monkeypatch.setattr(
        module,
        "validate_outbound_url",
        AsyncMock(return_value=[ipaddress.ip_address("8.8.8.8"), ipaddress.ip_address("1.1.1.1")]),
    )
    backend = module.PublicNetworkBackend()
    timeouts = []

    async def connect(*args, timeout, **kwargs):
        timeouts.append(timeout)
        await asyncio.sleep(0.01)
        raise httpcore.ConnectTimeout()

    monkeypatch.setattr(backend.backend, "connect_tcp", connect)
    with pytest.raises(httpcore.ConnectTimeout):
        await backend.connect_tcp("public.example", 443, timeout=1)
    assert 0 < timeouts[1] < timeouts[0] <= 1


def test_explicit_ca_bundle_is_used_without_disabling_verification(monkeypatch):
    import ssl
    from unittest.mock import Mock

    from cognee.tasks.web_scraper import public_http_transport as module

    context = ssl.create_default_context()
    create = Mock(return_value=context)
    monkeypatch.setattr(module.ssl, "create_default_context", create)
    transport = PublicHTTPTransport(ca_bundle="/explicit/ca.pem")
    create.assert_called_once_with(cafile="/explicit/ca.pem")
    assert transport._pool._ssl_context is context
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED


def test_explicit_proxy_is_used_only_when_selected():
    transport = PublicHTTPTransport(proxy_url="http://127.0.0.1:8080")
    assert isinstance(transport._pool, httpcore.AsyncHTTPProxy)
    assert transport._pool._proxy_url.host == b"127.0.0.1"


@pytest.mark.asyncio
async def test_concurrent_first_requests_share_one_client(monkeypatch):
    import asyncio
    from unittest.mock import Mock

    from cognee.tasks.web_scraper import public_http_transport as module
    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    factory = Mock(wraps=module.PublicHTTPTransport)
    monkeypatch.setattr(module, "PublicHTTPTransport", factory)
    crawler = DefaultUrlCrawler()
    try:
        await asyncio.gather(*(crawler._ensure_client() for _ in range(10)))
        factory.assert_called_once()
    finally:
        await crawler.close()


@pytest.mark.parametrize(
    "ipv4", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "100.64.0.1"]
)
def test_nat64_cannot_hide_a_blocked_ipv4(ipv4):
    translated = ipaddress.IPv6Address(
        int(ipaddress.IPv6Address("64:ff9b::")) | int(ipaddress.IPv4Address(ipv4))
    )
    assert ssrf._is_blocked_ip(translated)


def test_nat64_public_ipv4_remains_public():
    assert not ssrf._is_blocked_ip(ipaddress.ip_address("64:ff9b::5db8:d822"))


@pytest.mark.asyncio
async def test_browser_route_waiting_for_slot_obeys_navigation_deadline():
    import asyncio
    import time
    from types import SimpleNamespace

    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    route = SimpleNamespace(request=SimpleNamespace(resource_type="fetch"), abort=AsyncMock())
    with pytest.raises(TimeoutError):
        await DefaultUrlCrawler()._fetch_browser_route(
            route, semaphore=asyncio.Semaphore(0), deadline=time.monotonic() + 0.01
        )
    route.abort.assert_awaited_once_with("blockedbyclient")


@pytest.mark.asyncio
async def test_playwright_serialization_runs_off_caller_loop(monkeypatch):
    import threading

    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    caller = threading.get_ident()

    async def render(self, *args):
        assert threading.get_ident() != caller
        return "rendered"

    monkeypatch.setattr(DefaultUrlCrawler, "_render_browser", render)
    assert await DefaultUrlCrawler()._render_with_playwright("https://example.test") == "rendered"


@pytest.mark.parametrize("address", ["::ffff:127.0.0.1", "2002:7f00:1::", "2002:a9fe:a9fe::"])
def test_tunneled_ipv6_cannot_hide_non_public_ipv4(address):
    assert ssrf._is_blocked_ip(ipaddress.ip_address(address))

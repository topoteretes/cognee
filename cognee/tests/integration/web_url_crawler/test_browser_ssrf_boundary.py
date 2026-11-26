"""Real Chromium must not reach a local listener through a public redirect."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_chromium_redirect_never_connects_to_loopback(monkeypatch):
    playwright = pytest.importorskip("playwright.async_api")
    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    async with playwright.async_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed; run playwright install chromium")

    hits = []

    class Canary(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"private")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Canary)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    requests = []

    def public_response(request):
        requests.append(str(request.url))
        assert request.url.host == "public.example"
        return httpx.Response(
            302, headers={"Location": f"http://127.0.0.1:{server.server_port}/canary"}
        )

    async def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(public_response))

    monkeypatch.setattr(DefaultUrlCrawler, "_ensure_client", client)
    try:
        async with DefaultUrlCrawler(max_retries=0) as crawler:
            with pytest.raises(playwright.Error, match="ERR_BLOCKED_BY_CLIENT"):
                await crawler._render_with_playwright("https://public.example/redirect", js_wait=0)
        assert requests == ["https://public.example/redirect"]
        assert hits == []
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_many_subresources_render_with_independent_route_concurrency(monkeypatch):
    playwright = pytest.importorskip("playwright.async_api")
    from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler

    async with playwright.async_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed; run playwright install chromium")
    active = peak = 0

    async def public_response(request):
        nonlocal active, peak
        if request.url.path == "/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    "<script>window.loaded = 0</script>"
                    + "".join(f'<script async src="/part-{i}.js"></script>' for i in range(30))
                ),
            )
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.1)
            return httpx.Response(
                200,
                headers={"content-type": "text/javascript"},
                text="window.loaded++;document.title=String(window.loaded)",
            )
        finally:
            active -= 1

    async def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(public_response))

    monkeypatch.setattr(DefaultUrlCrawler, "_ensure_client", client)
    async with DefaultUrlCrawler(
        concurrency=1, browser_route_concurrency=10, timeout=3, max_retries=0
    ) as crawler:
        html = await crawler._render_with_playwright("https://public.example/", js_wait=0)
    assert "<title>30</title>" in html
    assert 1 < peak <= 10

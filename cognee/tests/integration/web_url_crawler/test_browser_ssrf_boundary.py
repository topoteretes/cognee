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

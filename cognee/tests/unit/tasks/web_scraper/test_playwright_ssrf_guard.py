"""Unit tests for the SSRF guard on DefaultUrlCrawler's Playwright path.

The Playwright objects are faked, so no browser is needed. The fake page asks the
context's route handler about every request it makes, including each redirect hop
and subresource, and behaves like Chromium when a route is aborted.
"""

from dataclasses import dataclass, field
from urllib.parse import urlparse

import pytest

from cognee.tasks.web_scraper import default_url_crawler as crawler_module
from cognee.tasks.web_scraper.default_url_crawler import DefaultUrlCrawler
from cognee.tasks.web_scraper.ssrf_protection import SSRFProtectionError

PUBLIC = "https://public.example"
INTERNAL = "http://127.0.0.1:8080"


@dataclass
class FakeResponse:
    status: int = 200
    location: str | None = None
    body: str = ""
    subresources: list[str] = field(default_factory=list)

    @property
    def headers(self) -> dict[str, str]:
        return {"location": self.location} if self.location else {}


SITE = {
    f"{PUBLIC}/start": FakeResponse(status=302, location=f"{INTERNAL}/internal"),
    f"{PUBLIC}/hop": FakeResponse(status=302, location=f"{PUBLIC}/page"),
    f"{PUBLIC}/page": FakeResponse(body="<p>public page</p>"),
    f"{PUBLIC}/with-internal-image": FakeResponse(
        body="<p>public page</p>", subresources=[f"{INTERNAL}/img.png"]
    ),
    f"{INTERNAL}/internal": FakeResponse(body="<p>internal page</p>"),
    f"{INTERNAL}/img.png": FakeResponse(body="png"),
}


class FakePlaywrightError(Exception):
    """Stands in for playwright.async_api.Error."""


class FakeRequest:
    def __init__(self, url: str, navigation: bool):
        self.url = url
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class FakeRoute:
    def __init__(self, request: FakeRequest):
        self.request = request
        self.response: FakeResponse | None = None
        self.outcome: str | None = None

    async def fetch(self, max_redirects=None) -> FakeResponse:
        assert max_redirects == 0, "redirects must be validated, not followed by fetch"
        return SITE[self.request.url]

    async def abort(self, error_code=None) -> None:
        self.outcome = f"abort:{error_code}"

    async def fulfill(self, response=None) -> None:
        self.outcome = "fulfill"
        self.response = response


class FakePage:
    def __init__(self, context: "FakeContext"):
        self._context = context
        self._html = ""

    async def _request(self, url: str, navigation: bool) -> FakeRoute:
        route = FakeRoute(FakeRequest(url, navigation))
        self._context.routes.append(route)
        if self._context.handler is None:
            # No interception: the browser fetches the URL itself.
            await route.fulfill(response=SITE[url])
        else:
            await self._context.handler(route)
        return route

    async def goto(self, url: str, wait_until=None, timeout=None) -> None:
        while True:
            route = await self._request(url, navigation=True)
            if route.outcome != "fulfill":
                raise FakePlaywrightError(f"net::ERR_BLOCKED_BY_CLIENT at {url}")
            if route.response.location is None:
                break
            url = route.response.location
        self._html = route.response.body
        for subresource in route.response.subresources:
            await self._request(subresource, navigation=False)

    async def content(self) -> str:
        return self._html


class FakeContext:
    def __init__(self, options: dict):
        self.options = options
        self.handler = None
        self.routes: list[FakeRoute] = []

    async def route(self, pattern: str, handler) -> None:
        assert pattern == "**/*"
        self.handler = handler

    async def new_page(self) -> FakePage:
        return FakePage(self)


class FakeBrowser:
    def __init__(self, playwright: "FakePlaywright"):
        self._playwright = playwright

    async def new_context(self, **options) -> FakeContext:
        context = FakeContext(options)
        self._playwright.contexts.append(context)
        return context

    async def close(self) -> None:
        pass


class FakePlaywright:
    def __init__(self):
        self.chromium = self
        self.launches = 0
        self.contexts: list[FakeContext] = []
        self.validated: list[str] = []

    async def launch(self, headless=True) -> FakeBrowser:
        self.launches += 1
        return FakeBrowser(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    @property
    def routes(self) -> list[FakeRoute]:
        return [route for context in self.contexts for route in context.routes]


@pytest.fixture
def fake_playwright(monkeypatch) -> FakePlaywright:
    fake = FakePlaywright()

    async def fake_validate(url: str) -> None:
        fake.validated.append(url)
        if urlparse(url).hostname == "127.0.0.1":
            raise SSRFProtectionError(f"URL host for {url} is blocked.")

    monkeypatch.setattr(crawler_module, "async_playwright", lambda: fake)
    monkeypatch.setattr(crawler_module, "PlaywrightError", FakePlaywrightError, raising=False)
    monkeypatch.setattr(crawler_module, "validate_outbound_url", fake_validate)
    return fake


def _crawler() -> DefaultUrlCrawler:
    return DefaultUrlCrawler(max_retries=2, retry_delay_factor=0)


@pytest.mark.asyncio
async def test_redirect_to_blocked_host_raises_and_is_not_retried(fake_playwright):
    with pytest.raises(SSRFProtectionError):
        await _crawler()._render_with_playwright(f"{PUBLIC}/start", js_wait=0)

    assert fake_playwright.launches == 1
    assert [route.request.url for route in fake_playwright.routes] == [f"{PUBLIC}/start"]
    assert fake_playwright.routes[0].outcome == "abort:blockedbyclient"
    assert f"{INTERNAL}/internal" in fake_playwright.validated


@pytest.mark.asyncio
async def test_allowed_redirect_is_followed_and_every_hop_is_validated(fake_playwright):
    html = await _crawler()._render_with_playwright(f"{PUBLIC}/hop", js_wait=0)

    assert html == "<p>public page</p>"
    requested = [route.request.url for route in fake_playwright.routes]
    assert requested == [f"{PUBLIC}/hop", f"{PUBLIC}/page"]
    assert all(url in fake_playwright.validated for url in requested)
    assert [route.outcome for route in fake_playwright.routes] == ["fulfill", "fulfill"]
    assert fake_playwright.contexts[0].options.get("service_workers") == "block"


@pytest.mark.asyncio
async def test_blocked_subresource_is_aborted_and_page_still_renders(fake_playwright):
    html = await _crawler()._render_with_playwright(f"{PUBLIC}/with-internal-image", js_wait=0)

    assert html == "<p>public page</p>"
    outcomes = {route.request.url: route.outcome for route in fake_playwright.routes}
    assert outcomes == {
        f"{PUBLIC}/with-internal-image": "fulfill",
        f"{INTERNAL}/img.png": "abort:blockedbyclient",
    }

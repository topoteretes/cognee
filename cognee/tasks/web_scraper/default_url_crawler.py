import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Any, Union, List, Dict, Optional
from urllib.parse import urlparse
import httpx

from cognee.shared.logging_utils import get_logger
from cognee.tasks.web_scraper.types import UrlsToHtmls

logger = get_logger()

try:
    from protego import Protego
except ImportError:
    logger.warning("Failed to import protego, make sure to install using pip install protego>=0.1")
    Protego = None

try:
    from playwright.async_api import async_playwright
except ImportError:
    logger.warning(
        "Failed to import playwright, make sure to install using pip install playwright>=1.9.0"
    )
    async_playwright = None


@dataclass
class RobotsTxtCache:
    """Cache for robots.txt data.

    Attributes:
        protego: Parsed robots.txt object (Protego instance).
        crawl_delay: Delay between requests (in seconds).
        timestamp: Time when the cache entry was created.
    """

    protego: Any
    crawl_delay: float
    timestamp: float = field(default_factory=time.time)


class DefaultUrlCrawler:
    def __init__(
        self,
        *,
        concurrency: int = 5,
        crawl_delay: float = 0.5,
        max_crawl_delay: Optional[float] = 10.0,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_delay_factor: float = 0.5,
        headers: Optional[Dict[str, str]] = None,
        robots_cache_ttl: float = 3600.0,
    ):
        """Initialize the BeautifulSoupCrawler.

        Args:
            concurrency: Number of concurrent requests allowed.
            crawl_delay: Minimum seconds between requests to the same domain.
            max_crawl_delay: Maximum crawl delay to respect from robots.txt (None = no limit).
            timeout: Per-request timeout in seconds.
            max_retries: Number of retries for failed requests.
            retry_delay_factor: Multiplier for exponential backoff on retries.
            headers: HTTP headers for requests (defaults to User-Agent: Cognee-Scraper/1.0).
            robots_cache_ttl: Time-to-live for robots.txt cache in seconds.
        """
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.crawl_delay = crawl_delay
        self.max_crawl_delay = max_crawl_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_factor = retry_delay_factor
        self.headers = headers or {"User-Agent": "Cognee-Scraper/1.0"}
        self.robots_cache_ttl = robots_cache_ttl
        self._last_request_time_per_domain: Dict[str, float] = {}
        self._robots_cache: Dict[str, RobotsTxtCache] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._robots_lock = asyncio.Lock()

    async def _ensure_client(self):
        """Initialize the HTTP client if not already created."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Enter the context manager, initializing the HTTP client."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, closing the HTTP client."""
        await self.close()

    @lru_cache(maxsize=1024)
    def _domain_from_url(self, url: str) -> str:
        """Extract the domain (netloc) from a URL.

        Args:
            url: The URL to parse.

        Returns:
            str: The domain (netloc) of the URL.
        """
        try:
            return urlparse(url).netloc
        except Exception:
            return url

    @lru_cache(maxsize=1024)
    def _get_domain_root(self, url: str) -> str:
        """Get the root URL (scheme and netloc) from a URL.

        Args:
            url: The URL to parse.

        Returns:
            str: The root URL (e.g., "https://example.com").
        """
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _respect_rate_limit(self, url: str, crawl_delay: Optional[float] = None):
        """Enforce rate limiting for requests to the same domain.

        Args:
            url: The URL to check.
            crawl_delay: Custom crawl delay in seconds (if any).
        """
        domain = self._domain_from_url(url)
        last = self._last_request_time_per_domain.get(domain)
        delay = crawl_delay if crawl_delay is not None else self.crawl_delay

        if last is None:
            self._last_request_time_per_domain[domain] = time.time()
            return

        elapsed = time.time() - last
        wait_for = delay - elapsed
        if wait_for > 0:
            logger.info(
                f"Rate limiting: waiting {wait_for:.2f}s before requesting {url} (crawl_delay={delay}s from robots.txt)"
            )
            await asyncio.sleep(wait_for)
            logger.info(f"Rate limit wait completed for {url}")
        self._last_request_time_per_domain[domain] = time.time()

    async def _get_robots_cache(self, domain_root: str) -> Optional[RobotsTxtCache]:
        """Get cached robots.txt data if valid.

        Args:
            domain_root: The root URL (e.g., "https://example.com").

        Returns:
            Optional[RobotsTxtCache]: Cached robots.txt data or None if expired or not found.
        """
        if Protego is None:
            return None

        cached = self._robots_cache.get(domain_root)
        if cached and (time.time() - cached.timestamp) < self.robots_cache_ttl:
            return cached
        return None

    async def _fetch_and_cache_robots(self, domain_root: str) -> RobotsTxtCache:
        """Fetch and cache robots.txt data.

        Args:
            domain_root: The root URL (e.g., "https://example.com").

        Returns:
            RobotsTxtCache: Cached robots.txt data with crawl delay.

        Raises:
            Exception: If fetching robots.txt fails.
        """
        async with self._robots_lock:
            cached = await self._get_robots_cache(domain_root)
            if cached:
                return cached

            robots_url = f"{domain_root}/robots.txt"
            try:
                await self._ensure_client()
                await self._respect_rate_limit(robots_url, self.crawl_delay)
                resp = await self._client.get(robots_url, timeout=5.0)
                content = resp.text if resp.status_code == 200 else ""
            except Exception as e:
                logger.debug(f"Failed to fetch robots.txt from {domain_root}: {e}")
                content = ""

            protego = Protego.parse(content) if content.strip() else None
            agent = next((v for k, v in self.headers.items() if k.lower() == "user-agent"), "*")

            crawl_delay = self.crawl_delay
            if protego:
                delay = protego.crawl_delay(agent) or protego.crawl_delay("*")
                if delay:
                    # Apply max_crawl_delay cap if configured
                    if self.max_crawl_delay is not None and delay > self.max_crawl_delay:
                        logger.warning(
                            f"robots.txt specifies crawl_delay={delay}s for {domain_root}, "
                            f"capping to max_crawl_delay={self.max_crawl_delay}s"
                        )
                        crawl_delay = self.max_crawl_delay
                    else:
                        crawl_delay = delay

            cache_entry = RobotsTxtCache(protego=protego, crawl_delay=crawl_delay)
            self._robots_cache[domain_root] = cache_entry
            return cache_entry

    async def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed by robots.txt.

        Args:
            url: The URL to check.

        Returns:
            bool: True if the URL is allowed, False otherwise.
        """
        if Protego is None:
            return True

        try:
            domain_root = self._get_domain_root(url)
            cache = await self._get_robots_cache(domain_root)
            if cache is None:
                cache = await self._fetch_and_cache_robots(domain_root)

            if cache.protego is None:
                return True

            agent = next((v for k, v in self.headers.items() if k.lower() == "user-agent"), "*")
            return cache.protego.can_fetch(agent, url) or cache.protego.can_fetch("*", url)
        except Exception as e:
            logger.debug(f"Error checking robots.txt for {url}: {e}")
            return True

    async def _get_crawl_delay(self, url: str) -> float:
        """Get the crawl delay for a URL from robots.txt.

        Args:
            url: The URL to check.

        Returns:
            float: Crawl delay in seconds.
        """
        if Protego is None:
            return self.crawl_delay

        try:
            domain_root = self._get_domain_root(url)
            cache = await self._get_robots_cache(domain_root)
            if cache is None:
                cache = await self._fetch_and_cache_robots(domain_root)
            return cache.crawl_delay
        except Exception:
            return self.crawl_delay

    async def _fetch_httpx(self, url: str) -> str:
        """Fetch a URL using HTTPX with retries.

        Args:
            url: The URL to fetch.

        Returns:
            str: The HTML content of the page.

        Raises:
            Exception: If all retry attempts fail.
        """
        await self._ensure_client()
        assert self._client is not None, "HTTP client not initialized"

        attempt = 0
        crawl_delay = await self._get_crawl_delay(url)
        logger.info(f"Fetching URL with httpx (crawl_delay={crawl_delay}s): {url}")

        while True:
            try:
                await self._respect_rate_limit(url, crawl_delay)
                resp = await self._client.get(url)
                resp.raise_for_status()
                logger.info(
                    f"Successfully fetched {url} (status={resp.status_code}, size={len(resp.text)} bytes)"
                )
                return resp.text
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"Fetch failed for {url} after {attempt} attempts: {exc}")
                    raise

                delay = self.retry_delay_factor * (2 ** (attempt - 1))
                logger.warning(
                    f"Retrying {url} after {delay:.2f}s (attempt {attempt}) due to {exc}"
                )
                await asyncio.sleep(delay)

    async def _render_with_playwright(
        self, url: str, js_wait: float = 1.0, timeout: Optional[float] = None
    ) -> str:
        """Fetch and render a URL using Playwright for JavaScript content.

        Args:
            url: The URL to fetch.
            js_wait: Seconds to wait for JavaScript to load.
            timeout: Timeout for the request (in seconds, defaults to instance timeout).

        Returns:
            str: The rendered HTML content.

        Raises:
            RuntimeError: If Playwright is not installed.
            Exception: If all retry attempts fail.
        """
        if async_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Install with `pip install playwright` and run `playwright install`."
            )

        timeout_val = timeout or self.timeout
        logger.info(
            f"Rendering URL with Playwright (js_wait={js_wait}s, timeout={timeout_val}s): {url}"
        )

        attempt = 0
        while True:
            try:
                async with async_playwright() as p:
                    logger.info(f"Launching headless Chromium browser for {url}")
                    browser = await p.chromium.launch(headless=True)
                    try:
                        context = await browser.new_context()
                        page = await context.new_page()
                        logger.info(f"Navigating to {url} and waiting for network idle")
                        await page.goto(
                            url,
                            wait_until="networkidle",
                            timeout=int(timeout_val * 1000),
                        )
                        if js_wait:
                            logger.info(f"Waiting {js_wait}s for JavaScript to execute")
                            await asyncio.sleep(js_wait)
                        content = await page.content()
                        logger.info(
                            f"Successfully rendered {url} with Playwright (size={len(content)} bytes)"
                        )
                        return content
                    finally:
                        await browser.close()
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"Playwright fetch failed for {url}: {exc}")
                    raise
                backoff = self.retry_delay_factor * (2 ** (attempt - 1))
                logger.warning(
                    f"Retrying playwright fetch {url} after {backoff:.2f}s (attempt {attempt})"
                )
                await asyncio.sleep(backoff)

    async def fetch_urls(
        self,
        urls: Union[str, List[str]],
        *,
        use_playwright: bool = False,
        playwright_js_wait: float = 0.8,
    ) -> UrlsToHtmls:
        """Fetch and extract content from URLs using BeautifulSoup or Playwright.

        Args:
            urls: A single URL, list of URLs, or dict mapping URLs to extraction rules.
            extraction_rules: Default extraction rules for string or list URLs.
            use_playwright: If True, use Playwright for JavaScript rendering.
            playwright_js_wait: Seconds to wait for JavaScript to load.
            join_all_matches: If True, extract all matching elements for each rule.

        Returns:
            Dict[str, str]: A dictionary mapping URLs to their extracted content.

        Raises:
            ValueError: If extraction_rules are missing when required or if urls is invalid.
            Exception: If fetching or extraction fails.
        """
        if isinstance(urls, str):
            urls = [urls]
        else:
            raise ValueError(f"Invalid urls type: {type(urls)}")

        async def _task(url: str):
            async with self._sem:
                try:
                    logger.info(f"Processing URL: {url}")

                    # Check robots.txt
                    allowed = await self._is_url_allowed(url)
                    if not allowed:
                        logger.warning(f"URL disallowed by robots.txt: {url}")
                        return url, ""

                    logger.info(f"Robots.txt check passed for {url}")

                    # Fetch HTML
                    if use_playwright:
                        logger.info(
                            f"Rendering {url} with Playwright (JS wait: {playwright_js_wait}s)"
                        )
                        html = await self._render_with_playwright(
                            url, js_wait=playwright_js_wait, timeout=self.timeout
                        )
                    else:
                        logger.info(f"Fetching {url} with httpx")
                        html = await self._fetch_httpx(url)

                    logger.info(f"Successfully fetched HTML from {url} ({len(html)} bytes)")

                    return url, html

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    return url, ""

        logger.info(f"Creating {len(urls)} async tasks for concurrent fetching")
        tasks = [asyncio.create_task(_task(u)) for u in urls]
        results = {}
        completed = 0
        total = len(tasks)

        for coro in asyncio.as_completed(tasks):
            url, html = await coro
            results[url] = html
            completed += 1
            logger.info(f"Progress: {completed}/{total} URLs processed")

        logger.info(f"Completed fetching all {len(results)} URL(s)")
        return results

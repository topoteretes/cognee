"""BeautifulSoup-based web crawler for extracting content from web pages.

This module provides the BeautifulSoupCrawler class for fetching and extracting content
from web pages using BeautifulSoup or Playwright for JavaScript-rendered pages. It
supports robots.txt handling, rate limiting, and custom extraction rules.
"""

import asyncio
import time
from typing import Union, List, Dict, Any, Optional
from urllib.parse import urlparse
from dataclasses import dataclass, field
from functools import lru_cache
import httpx
from bs4 import BeautifulSoup
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from playwright.async_api import async_playwright
except ImportError:
    logger.error(
        "Failed to import playwright, make sure to install using pip install playwright>=1.9.0"
    )
    async_playwright = None

try:
    from protego import Protego
except ImportError:
    logger.error("Failed to import protego, make sure to install using pip install protego>=0.1")
    Protego = None


@dataclass
class ExtractionRule:
    """Normalized extraction rule for web content.

    Attributes:
        selector: CSS selector for extraction (if any).
        xpath: XPath expression for extraction (if any).
        attr: HTML attribute to extract (if any).
        all: If True, extract all matching elements; otherwise, extract first.
        join_with: String to join multiple extracted elements.
    """

    selector: Optional[str] = None
    xpath: Optional[str] = None
    attr: Optional[str] = None
    all: bool = False
    join_with: str = " "


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


class BeautifulSoupCrawler:
    """Crawler for fetching and extracting web content using BeautifulSoup.

    Supports asynchronous HTTP requests, Playwright for JavaScript rendering, robots.txt
    compliance, and rate limiting. Extracts content using CSS selectors or XPath rules.

    Attributes:
        concurrency: Number of concurrent requests allowed.
        crawl_delay: Minimum seconds between requests to the same domain.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries for failed requests.
        retry_delay_factor: Multiplier for exponential backoff on retries.
        headers: HTTP headers for requests (e.g., User-Agent).
        robots_cache_ttl: Time-to-live for robots.txt cache in seconds.
    """

    def __init__(
        self,
        *,
        concurrency: int = 5,
        crawl_delay: float = 0.5,
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
            timeout: Per-request timeout in seconds.
            max_retries: Number of retries for failed requests.
            retry_delay_factor: Multiplier for exponential backoff on retries.
            headers: HTTP headers for requests (defaults to User-Agent: Cognee-Scraper/1.0).
            robots_cache_ttl: Time-to-live for robots.txt cache in seconds.
        """
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.crawl_delay = crawl_delay
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
            await asyncio.sleep(wait_for)
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
                crawl_delay = delay if delay else self.crawl_delay

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

        while True:
            try:
                await self._respect_rate_limit(url, crawl_delay)
                resp = await self._client.get(url)
                resp.raise_for_status()
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
        attempt = 0
        while True:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    try:
                        context = await browser.new_context()
                        page = await context.new_page()
                        await page.goto(
                            url,
                            wait_until="networkidle",
                            timeout=int((timeout or self.timeout) * 1000),
                        )
                        if js_wait:
                            await asyncio.sleep(js_wait)
                        return await page.content()
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

    def _normalize_rule(self, rule: Union[str, Dict[str, Any]]) -> ExtractionRule:
        """Normalize an extraction rule to an ExtractionRule dataclass.

        Args:
            rule: A string (CSS selector) or dict with extraction parameters.

        Returns:
            ExtractionRule: Normalized extraction rule.

        Raises:
            ValueError: If the rule is invalid.
        """
        if isinstance(rule, str):
            return ExtractionRule(selector=rule)
        if isinstance(rule, dict):
            return ExtractionRule(
                selector=rule.get("selector"),
                xpath=rule.get("xpath"),
                attr=rule.get("attr"),
                all=bool(rule.get("all", False)),
                join_with=rule.get("join_with", " "),
            )
        raise ValueError(f"Invalid extraction rule: {rule}")

    def _extract_with_bs4(self, html: str, rule: ExtractionRule) -> str:
        """Extract content from HTML using BeautifulSoup or lxml XPath.

        Args:
            html: The HTML content to extract from.
            rule: The extraction rule to apply.

        Returns:
            str: The extracted content.

        Raises:
            RuntimeError: If XPath is used but lxml is not installed.
        """
        soup = BeautifulSoup(html, "html.parser")

        if rule.xpath:
            try:
                from lxml import html as lxml_html
            except ImportError:
                raise RuntimeError(
                    "XPath requested but lxml is not available. Install lxml or use CSS selectors."
                )
            doc = lxml_html.fromstring(html)
            nodes = doc.xpath(rule.xpath)
            texts = []
            for n in nodes:
                if hasattr(n, "text_content"):
                    texts.append(n.text_content().strip())
                else:
                    texts.append(str(n).strip())
            return rule.join_with.join(t for t in texts if t)

        if not rule.selector:
            return ""

        if rule.all:
            nodes = soup.select(rule.selector)
            pieces = []
            for el in nodes:
                if rule.attr:
                    val = el.get(rule.attr)
                    if val:
                        pieces.append(val.strip())
                else:
                    text = el.get_text(strip=True)
                    if text:
                        pieces.append(text)
            return rule.join_with.join(pieces).strip()
        else:
            el = soup.select_one(rule.selector)
            if el is None:
                return ""
            if rule.attr:
                val = el.get(rule.attr)
                return (val or "").strip()
            return el.get_text(strip=True)

    async def fetch_with_bs4(
        self,
        urls: Union[str, List[str], Dict[str, Dict[str, Any]]],
        extraction_rules: Optional[Dict[str, Any]] = None,
        *,
        use_playwright: bool = False,
        playwright_js_wait: float = 0.8,
        join_all_matches: bool = False,
    ) -> Dict[str, str]:
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
        url_rules_map: Dict[str, Dict[str, Any]] = {}

        if isinstance(urls, str):
            if not extraction_rules:
                raise ValueError("extraction_rules required when urls is a string")
            url_rules_map[urls] = extraction_rules
        elif isinstance(urls, list):
            if not extraction_rules:
                raise ValueError("extraction_rules required when urls is a list")
            for url in urls:
                url_rules_map[url] = extraction_rules
        elif isinstance(urls, dict):
            url_rules_map = urls
        else:
            raise ValueError(f"Invalid urls type: {type(urls)}")

        normalized_url_rules: Dict[str, List[ExtractionRule]] = {}
        for url, rules in url_rules_map.items():
            normalized_rules = []
            for _, rule in rules.items():
                r = self._normalize_rule(rule)
                if join_all_matches:
                    r.all = True
                normalized_rules.append(r)
            normalized_url_rules[url] = normalized_rules

        async def _task(url: str):
            async with self._sem:
                try:
                    allowed = await self._is_url_allowed(url)
                    if not allowed:
                        logger.warning(f"URL disallowed by robots.txt: {url}")
                        return url, ""

                    if use_playwright:
                        html = await self._render_with_playwright(
                            url, js_wait=playwright_js_wait, timeout=self.timeout
                        )
                    else:
                        html = await self._fetch_httpx(url)

                    pieces = []
                    for rule in normalized_url_rules[url]:
                        text = self._extract_with_bs4(html, rule)
                        if text:
                            pieces.append(text)

                    concatenated = " ".join(pieces).strip()
                    return url, concatenated

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    return url, ""

        tasks = [asyncio.create_task(_task(u)) for u in url_rules_map.keys()]
        results = {}

        for coro in asyncio.as_completed(tasks):
            url, text = await coro
            results[url] = text

        return results
